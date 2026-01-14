#ifndef SHARED_CUH
#define SHARED_CUH

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

// Citire matrice din fisier
template <typename T>
void citire_matrice_dinamica(T* &ptr, int n, int m, std::ifstream& fin) {
    ptr = new T[n * m];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            fin >> ptr[i * m + j];
    }
}

// Eliberare memorie
template <typename T>
void eliberare_memorie(T* &ptr) {
    delete[] ptr;
    ptr = nullptr;
}

// Scriere matrice in fisier
template <typename T>
void scrie_matrice_dinamica(T* &ptr, int n, int m, std::ostream& out) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out << std::fixed << std::setprecision(6) << ptr[i * m + j] << ' ';
        }
        out << '\n';
    }
}

// Testare rezultate - compara doua matrice
template <typename T>
void testeaza_rezultate_dinamic(T* &ptr, int n, int m, T* &ptr_test) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            T diff = ptr_test[i * m + j] - ptr[i * m + j];
            if (diff < 0) diff = -diff;
            if (diff > 0.0001f) {
                std::cerr << std::fixed << std::setprecision(6) 
                    << "Diferenta la elementul " << i << ", " << j 
                    << ": " << ptr[i * m + j] << " != " << ptr_test[i * m + j] << "\n";
                throw std::runtime_error("Test failed!");
            }
        }
    }
}

// CUDA error checking macro
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

#endif
