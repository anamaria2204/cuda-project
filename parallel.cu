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

__global__ void convolution_strip_kernel(int *f, int *c, int n, int m, int k, 
                                         int *prev_row_buffers,  
                                         int *temp_row_buffers,  
                                         int *halo_buffers)      
{
    int tid = threadIdx.x; 
    int total_threads = blockDim.x;

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
    
    for (int j = 0; j < m; ++j) {
        halo_buffers[tid * m + j] = f[start_row * m + j];
    }
    
    __syncthreads();

    int no_borders = 1;

    for (int i = start_row; i < end_row; ++i) {
        
        if (i == start_row && i > 0) {
             for(int j=0; j<m; ++j) my_prev_buffer[j] = f[(i-1) * m + j];
        }

        // Calcul convolutie pentru linia i
        for (int j = 0; j < m; ++j) {
            int suma = 0;

            // Iteram Kernel Linii (ki)
            for (int ki = -no_borders; ki <= no_borders; ++ki) {
                int x = i + ki; 
                int *source_row;
                
                // Cazul 1: Vecinul este 'i-1' (randul de sus)
                if (x < i) {
                    if (i == start_row) {
                        // La startul chunk-ului, luam din global (e safe, vecinul de sus e departe)
                        source_row = (x < 0) ? f : &f[x * m]; 
                    } else {
                        // In interiorul chunk-ului, luam din bufferul local
                        source_row = my_prev_buffer;
                    }
                } 
                // Cazul 2: Vecinul este 'i+1' (randul de jos)
                else if (x > i) {
                    int next_thread_start = end_row;
                    if (x == next_thread_start && (tid + 1) < total_threads) {
                        source_row = &halo_buffers[(tid + 1) * m]; // Citim din HALO vecin
                    } else {
                        source_row = &f[x * m]; 
                    }
                }
                // Cazul 3: Vecinul este 'i' (randul curent)
                else {
                    source_row = &f[i * m];
                }
                
                // Iteram Kernel Coloane (kj)
                for (int kj = -no_borders; kj <= no_borders; ++kj) {
                    int val;
                    
                    // Calculam indexul coloanei cu clamp
                    int y = j + kj;
                    int y_clamped = (y < 0) ? 0 : (y >= m ? m - 1 : y);
                    
                    // Verificam daca source_row este un buffer special sau pointer in f
                    bool is_buffer = (source_row == my_prev_buffer) || 
                                     (source_row >= halo_buffers && source_row < halo_buffers + total_threads * m);

                    if (is_buffer) {
                         // Daca e buffer, accesam direct indexul [y_clamped]
                         val = source_row[y_clamped];
                    } else {
                         // Daca e pointer in F global, trebuie sa avem grija la indexare
                         // source_row este deja &f[x*m] SAU f (daca x<0).
                         int x_clamped = (x < 0) ? 0 : (x >= n ? n - 1 : x);
                         val = f[x_clamped * m + y_clamped];
                    }
                    
                    suma += val * c[(ki + no_borders) * k + (kj + no_borders)];
                }
            }
            my_temp_buffer[j] = suma;
        }

        // Salvam linia curenta in prev pentru iteratia urmatoare
        for(int j=0; j<m; ++j) {
            my_prev_buffer[j] = f[i * m + j];
        }

        // Scriem rezultatul (In-Place)
        for(int j=0; j<m; ++j) {
            f[i * m + j] = my_temp_buffer[j];
        }
    }
}

void cerceteaza_paralel() {
    cudaMalloc(&f_device, n * m * sizeof(int));
    cudaMalloc(&c_device, k * k * sizeof(int));
    
    int *prev_row_buffers_device;
    cudaMalloc(&prev_row_buffers_device, p * m * sizeof(int));
    
    int *temp_row_buffers_device;
    cudaMalloc(&temp_row_buffers_device, p * m * sizeof(int));
    
    int *halo_buffers_device; 
    cudaMalloc(&halo_buffers_device, p * m * sizeof(int));

    cudaCheckError();

    cudaMemcpy(f_device, f_host, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_host, k * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    auto start_time = chrono::high_resolution_clock::now();

    convolution_strip_kernel<<<1, p>>>(
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