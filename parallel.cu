#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "shared.cuh"

using namespace std;

// Dimensiuni
int n, m;
int *f_host, *f_device;
int k;
int *c_host, *c_device;
int p; // thread-uri pe bloc

// Kernel: convolutie 3x3 in-place cu shared memory si block-based distribution
__global__ void convolution_row_kernel(int *f, int *c, int *prev_row_buffer, int n, int m, int k, 
                                       int row_idx, int *temp_row) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    extern __shared__ int shared_kernel[];
    
    // Incarca kernel 3x3 in shared memory
    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x) {
        shared_kernel[idx] = c[idx];
    }
    __syncthreads();

    int i = row_idx;
    int no_borders = 1;

    // Distribuire coloane: m/total_threads + rest in primii thread-uri
    int cols_per_thread = m / total_threads;
    int remainder = m % total_threads;

    int start, end;
    if (thread_id < remainder) {
        start = thread_id * (cols_per_thread + 1);
        end = start + cols_per_thread + 1;
    } else {
        start = remainder * (cols_per_thread + 1) + (thread_id - remainder) * cols_per_thread;
        end = start + cols_per_thread;
    }

    for (int j = start; j < end; j++) {
        int suma = 0;

        for (int ki = -no_borders; ki <= no_borders; ++ki) {
            int x = i + ki;
            int x_clamped = (x < 0) ? 0 : (x >= n ? n - 1 : x);

            int *source;

            // Dacă rândul clamped este < i, folosim buffer (valoarea originală)
            if (x_clamped < i) {
                source = prev_row_buffer;
            if (x_clamped < i) {
                source = prev_row_buffer;  // Randul anterior (original)
            } else {
                source = &f[x_clamped * m];  // Randul nemodificat   int y_clamped = (y < 0) ? 0 : (y >= m ? m - 1 : y);

                // Folosim kernelul din shared memory in loc din global memory
                suma += source[y_clamped] * shared_kernel[(ki + no_borders) * k + (kj + no_borders)];
            }
        temp_row[j] = suma;
    }
}

void cerceteaza_paralel() {
    // Aloca memoria pe device
    cudaMalloc(&f_device, n * m * sizeof(int));
    cudaMalloc(&c_device, k * k * sizeof(int));
    cudaCheckError();

    // Buffer temporar pentru randul curent (pe device)
    int *temre si copiere date GPU
    cudaMalloc(&f_device, n * m * sizeof(int));
    cudaMalloc(&c_device, k * k * sizeof(int));
    
    int *temp_row_device;
    cudaMalloc(&temp_row_device, m * sizeof(int));
    
    int *prev_row_buffer_device;
    cudaMalloc(&prev_row_buffer_device, m * sizeof(int));
    cudaCheckError();

    cudaMemcpy(f_device, f_host, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_host, k * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    auto start_time = chrono::high_resolution_clock::now();

    int threads_per_block = p;
    int blocks = 32;
    int total_threads = blocks * threads_per_block;
    int shared_mem_size = k * k * sizeof(int);

    cerr << "Configurare paralel: " << blocks << " blocuri x " << threads_per_block 
         << " thread-uri = " << total_threads << " thread-uri total\n";

    // Procesare rand cu rand (in-place)
    for (int i = 0; i < n; ++i) {
        convolution_row_kernel<<<blocks, threads_per_block, shared_mem_size>>>(f_device, c_device, prev_row_buffer_device, n, m, k, i, temp_row_device);
        cudaCheckError();
        cudaDeviceSynchronize();
        
        cudaMemcpy(prev_row_buffer_device, &f_device[i * m], m * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&f_device[i * m], temp_row_device, m * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaCheckError();
    }

    cudaDeviceSynchronize();
    auto end_time = chrono::high_resolution_clock::now();

    cudaMemcpy(f_host, f_device, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    chrono::duration<double, milli> delta = end_time - start_time;
    cout << fixed << setprecision(6) << delta.count();

    std::ofstream out("output_parallel.txt");
    scrie_matrice_dinamica(f_host, n, m, out);
    out.close();

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Utilizare: " << argv[0] << " <n> <m> <k> <p>\n";
        cerr << "  n, m: dimensiuni matrice\n";
        cerr << "  k: dimensiune kernel (3)\n";
        cerr << "  p: thread-uri pe bloc\n";
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

    // Citire matrice
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
