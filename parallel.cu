#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "shared.cuh"

using namespace std;

int n, m;
int *f_host, *f_device;
int k;
int *c_host, *c_device;
int p;

// --- KERNEL 1: Pregatire Halo (Salvam prima linie a fiecarui thread) ---
__global__ void fill_halo_kernel(int *f, int n, int m, int *halo_buffers) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Recalculam aceeasi distributie a liniilor
    int rows_per_thread = n / total_threads;
    int remainder = n % total_threads;
    int start_row;

    if (tid < remainder) {
        start_row = tid * (rows_per_thread + 1);
    } else {
        int offset = remainder * (rows_per_thread + 1);
        start_row = offset + (tid - remainder) * rows_per_thread;
    }

    if (start_row >= n) return;

    // Salvam halo (prima linie) in buffer global
    for (int j = 0; j < m; ++j) {
        halo_buffers[tid * m + j] = f[start_row * m + j];
    }
}

// --- KERNEL 2: Procesare Convolutie ---
__global__ void convolution_compute_kernel(int *f, int *c, int n, int m, int k, 
                                           int *prev_row_buffers,  
                                           int *temp_row_buffers,  
                                           int *halo_buffers)      
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    int rows_per_thread = n / total_threads;
    int remainder = n % total_threads;

    int start_row, end_row; 

    if (tid < remainder) {
        start_row = tid * (rows_per_thread + 1);
        end_row = start_row + rows_per_thread + 1;
    } else {
        int offset = remainder * (rows_per_thread + 1);
        start_row = offset + (tid - remainder) * rows_per_thread;
        end_row = start_row + rows_per_thread;
    }

    if (start_row >= n) return; 

    int *my_prev_buffer = &prev_row_buffers[tid * m]; 
    int *my_temp_buffer = &temp_row_buffers[tid * m];
    
    // NU mai avem nevoie de salvarea halo aici si nici de __syncthreads()
    // Deoarece Kernel 1 a garantat ca datele sunt acolo.

    int no_borders = 1;

    for (int i = start_row; i < end_row; ++i) {
        
        // Initializare buffer prev la prima linie din chunk
        if (i == start_row && i > 0) {
             for(int j=0; j<m; ++j) my_prev_buffer[j] = f[(i-1) * m + j];
        }

        // Calcul convolutie
        for (int j = 0; j < m; ++j) {
            int suma = 0;
            for (int ki = -no_borders; ki <= no_borders; ++ki) {
                int x = i + ki; 
                int *source_row;
                
                // Cazul 1: Vecinul este 'i-1'
                if (x < i) {
                    if (i == start_row) {
                        source_row = (x < 0) ? f : &f[x * m]; 
                    } else {
                        source_row = my_prev_buffer;
                    }
                } 
                // Cazul 2: Vecinul este 'i+1'
                else if (x > i) {
                    int next_thread_start = end_row;
                    // Aici e cheia: Citim din halo_buffers populate de Kernel 1
                    if (x == next_thread_start && (tid + 1) < total_threads) {
                        source_row = &halo_buffers[(tid + 1) * m]; 
                    } else {
                        source_row = &f[x * m]; 
                    }
                }
                // Cazul 3: Vecinul este 'i'
                else {
                    source_row = &f[i * m];
                }
                
                for (int kj = -no_borders; kj <= no_borders; ++kj) {
                    int val;
                    int y = j + kj;
                    int y_clamped = (y < 0) ? 0 : (y >= m ? m - 1 : y);
                    
                    bool is_buffer = (source_row == my_prev_buffer) || 
                                     (source_row >= halo_buffers && source_row < halo_buffers + total_threads * m);

                    if (is_buffer) {
                         val = source_row[y_clamped];
                    } else {
                         int x_clamped = (x < 0) ? 0 : (x >= n ? n - 1 : x);
                         val = f[x_clamped * m + y_clamped];
                    }
                    suma += val * c[(ki + no_borders) * k + (kj + no_borders)];
                }
            }
            my_temp_buffer[j] = suma;
        }

        // Update buffers
        for(int j=0; j<m; ++j) my_prev_buffer[j] = f[i * m + j];
        for(int j=0; j<m; ++j) f[i * m + j] = my_temp_buffer[j];
    }
}

void cerceteaza_paralel() {
    cudaMalloc(&f_device, n * m * sizeof(int));
    cudaMalloc(&c_device, k * k * sizeof(int));
    
    // CONFIGURARE BLOCURI
    int blocks = 8;
    int total_threads_global = blocks * p; // Numar total fire in tot gridul

    // ALOCARE CORECTA: Inmultim cu total_threads_global, nu doar cu p!
    int *prev_row_buffers_device;
    cudaMalloc(&prev_row_buffers_device, total_threads_global * m * sizeof(int));
    
    int *temp_row_buffers_device;
    cudaMalloc(&temp_row_buffers_device, total_threads_global * m * sizeof(int));
    
    int *halo_buffers_device; 
    cudaMalloc(&halo_buffers_device, total_threads_global * m * sizeof(int));

    cudaCheckError();

    cudaMemcpy(f_device, f_host, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_host, k * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    auto start_time = chrono::high_resolution_clock::now();

    cerr << "Configurare: " << blocks << " blocuri x " << p << " thread-uri. Total: " << total_threads_global << "\n";

    // PASUL 1: Umplem Halo
    fill_halo_kernel<<<blocks, p>>>(f_device, n, m, halo_buffers_device);
    cudaCheckError();
    
    // PASUL 2: Bariera Globala (CPU Sync)
    // Asigura ca toate blocurile au scris halo-ul inainte ca cineva sa citeasca
    cudaDeviceSynchronize(); 

    // PASUL 3: Calcul
    convolution_compute_kernel<<<blocks, p>>>(
        f_device, c_device, n, m, k, 
        prev_row_buffers_device, temp_row_buffers_device, halo_buffers_device
    );
    cudaCheckError();
    
    cudaDeviceSynchronize();
    auto end_time = chrono::high_resolution_clock::now();

    cudaMemcpy(f_host, f_device, n * m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();

    chrono::duration<double, milli> delta = end_time - start_time;
    cout << fixed << setprecision(6) << delta.count();

    std::ofstream out("output_parallel.txt");
    scrie_matrice_dinamica(f_host, n, m, out);
    out.close();

    cudaFree(f_device);
    cudaFree(c_device);
    cudaFree(prev_row_buffers_device);
    cudaFree(temp_row_buffers_device);
    cudaFree(halo_buffers_device);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Utilizare: " << argv[0] << " <n> <m> <k> <p>\n";
        return 1;
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    k = atoi(argv[3]);
    p = atoi(argv[4]);

    if (k != 3) {
        cerr << "Aceasta aplicatie este configurata pentru k=3.\n";
        return 1;
    }

    std::ifstream in("input.txt");
    if (!in.is_open()) {
        cerr << "Nu pot deschide input.txt\n";
        return 1;
    }
    citire_matrice_dinamica(f_host, n, m, in);
    citire_matrice_dinamica(c_host, k, k, in);
    in.close();

    cerceteaza_paralel();

    eliberare_memorie(f_host);
    eliberare_memorie(c_host);

    return 0;
}