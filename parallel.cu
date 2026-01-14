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

// Kernel pentru procesarea unui singur rând - in-place cu buffer pentru rândul anterior
// Fiecare thread proceseaza o pozitie (row, j) din randul respectiv
__global__ void convolution_row_kernel(int *f, int *c, int *prev_row_buffer, int n, int m, int k, 
                                       int row_idx, int *temp_row) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= m) return;

    int i = row_idx;
    int suma = 0;
    int no_borders = 1; // k == 3

    for (int ki = -no_borders; ki <= no_borders; ++ki) {
        int x = i + ki;
        int x_clamped = (x < 0) ? 0 : (x >= n ? n - 1 : x);

        int *source;

        // Dacă rândul clamped este < i, folosim buffer (valoarea originală)
        if (x_clamped < i) {
            source = prev_row_buffer;
        } else {
            // Altfel, folosim valorile din f (care sunt originale pentru rândurile neprocessate)
            source = &f[x_clamped * m];
        }

        for (int kj = -no_borders; kj <= no_borders; ++kj) {
            int y = j + kj;
            int y_clamped = (y < 0) ? 0 : (y >= m ? m - 1 : y);

            suma += source[y_clamped] * c[(ki + no_borders) * k + (kj + no_borders)];
        }
    }

    temp_row[j] = suma;
}

void cerceteaza_paralel() {
    // Aloca memoria pe device
    cudaMalloc(&f_device, n * m * sizeof(float));
    cudaMalloc(&c_device, k * k * sizeof(float));
    cudaCheckError();

    // Buffer temporar pentru randul curent (pe device)
    int *temp_row_device;
    cudaMalloc(&temp_row_device, m * sizeof(int));
    cudaCheckError();

    // Buffer pentru rândul anterior (valoarea originală)
    int *prev_row_buffer_device;
    cudaMalloc(&prev_row_buffer_device, m * sizeof(int));
    cudaCheckError();

    // Copie date pe device
    cudaMemcpy(f_device, f_host, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_host, k * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    auto start_time = chrono::high_resolution_clock::now();

    // Configurare blocuri pentru procesarea unui rând
    int threads_per_block = p; // parametru configurable
    int blocks = (m + threads_per_block - 1) / threads_per_block;

    // Afisare configuratie paralel
    int total_threads = blocks * threads_per_block;
    cerr << "Configurare paralel: " << blocks << " blocuri x " << threads_per_block 
         << " thread-uri = " << total_threads << " thread-uri total\n";

    // Procesare rând cu rând - IN-PLACE
    for (int i = 0; i < n; ++i) {
        // Calcul rândul i pe device - cu buffer pentru rândul anterior
        convolution_row_kernel<<<blocks, threads_per_block>>>(f_device, c_device, prev_row_buffer_device, n, m, k, i, temp_row_device);
        cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();

        // Salvez rândul original (i) în buffer (înainte de suprascriere)
        cudaMemcpy(prev_row_buffer_device, &f_device[i * m], m * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaCheckError();

        // Copiez rezultatul inapoi în f_device (in-place) - pentru rândul i
        cudaMemcpy(&f_device[i * m], temp_row_device, m * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaCheckError();
    }

    cudaDeviceSynchronize();
    cudaCheckError();

    auto end_time = chrono::high_resolution_clock::now();

    // Copie rezultatul final înapoi pe host
    cudaMemcpy(f_host, f_device, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    chrono::duration<double, milli> delta = end_time - start_time;
    cout << fixed << setprecision(6) << delta.count();

    // Salveaza rezultatul
    std::ofstream out("output_parallel.txt");
    scrie_matrice_dinamica(f_host, n, m, out);
    out.close();

    // Eliberare memorie device
    cudaFree(f_device);
    cudaFree(c_device);
    cudaFree(temp_row_device);
    cudaFree(prev_row_buffer_device);
}

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
