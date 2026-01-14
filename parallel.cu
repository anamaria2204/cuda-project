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

// Kernel cu distributie STATICA a coloanelor (Chunking)
__global__ void convolution_row_kernel(int *f, int *c, int *prev_row_buffer, int n, int m, int k, int row_idx, int *temp_row) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Incarcare kernel in shared memory
    extern __shared__ int shared_kernel[];
    int local_tid = threadIdx.x;
    for (int idx = local_tid; idx < k * k; idx += blockDim.x) {
        shared_kernel[idx] = c[idx];
    }
    __syncthreads();

    int i = row_idx;
    int no_borders = 1;

    // Calculam cate elemente (coloane) revin fiecarui thread
    int count = m / total_threads;
    int remainder = m % total_threads;

    int start, end;

    // Primele 'remainder' thread-uri primesc count + 1 elemente
    // Restul thread-urilor primesc count elemente
    if (tid < remainder) {
        start = tid * (count + 1);
        end = start + count + 1;
    } else {
        // Offset-ul este dat de cele 'remainder' thread-uri care au luat cate (count+1)
        // plus cele (tid - remainder) thread-uri dinaintea mea care au luat cate count
        int offset = remainder * (count + 1);
        start = offset + (tid - remainder) * count;
        end = start + count;
    }

    // Procesare interval continuu [start, end)
    for (int j = start; j < end; ++j) {
        if (j >= m) break; // Safety check limite

        int suma = 0;

        // Iterare vecini pe verticala (fereastra kernel)
        for (int ki = -no_borders; ki <= no_borders; ++ki) {
            int x = i + ki;
            int x_clamped = (x < 0) ? 0 : (x >= n ? n - 1 : x); // Clamp la margini (padding)

            int *source;
            // Daca accesam randul anterior (deja suprascris in f), citim varianta originala din buffer
            if (x_clamped < i) {
                source = prev_row_buffer;
            } else {
                source = &f[x_clamped * m];
            }

            // Iterare vecini pe orizontala
            for (int kj = -no_borders; kj <= no_borders; ++kj) {
                int y = j + kj;
                int y_clamped = (y < 0) ? 0 : (y >= m ? m - 1 : y);
                
                // Calcul convolutie folosind kernel-ul optimizat din Shared Memory
                suma += source[y_clamped] * shared_kernel[(ki + no_borders) * k + (kj + no_borders)];
            }
        }
        temp_row[j] = suma;
    }
}

void cerceteaza_paralel() {
    cudaMalloc(&f_device, n * m * sizeof(int));
    cudaMalloc(&c_device, k * k * sizeof(int));
    
    int *prev_row_buffer_device;
    cudaMalloc(&prev_row_buffer_device, m * sizeof(int));
    
    int *temp_row_device;
    cudaMalloc(&temp_row_device, m * sizeof(int));
    
    cudaCheckError();

    cudaMemcpy(f_device, f_host, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_host, k * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    auto start_time = chrono::high_resolution_clock::now();

    int threads_per_block = p;
    int blocks = 32;

    int shared_mem_size = k * k * sizeof(int);

    // Calculam total fire pentru a verifica distributia (optional debug)
    int total_threads = blocks * threads_per_block;
    cerr << "Configurare paralel: " << blocks << " blocuri x " << threads_per_block 
         << " thread-uri. Total fire: " << total_threads << ". Distributie statica pe coloane.\n";

    for (int i = 0; i < n; ++i) {
        convolution_row_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            f_device, c_device, prev_row_buffer_device, n, m, k, i, temp_row_device);
        cudaCheckError();
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(prev_row_buffer_device, &f_device[i * m], m * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&f_device[i * m], temp_row_device, m * sizeof(int), cudaMemcpyDeviceToDevice);
    }

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
    cudaFree(prev_row_buffer_device);
    cudaFree(temp_row_device);
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