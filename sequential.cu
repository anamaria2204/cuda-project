#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include "shared.cuh"

using namespace std;

// Dimensiuni matrice F(n, m)
int n, m;
int *f;

// Dimensiune matrice convolutie C(k, k)
int k;
int *c;

void calculeaza_convolutie_inplace() {
    const int no_borders = 1; // k == 3 => no_borders = 1

    // Buffer circular pentru ultimele no_borders randuri (= 1)
    int** buffer = new int*[no_borders];
    for (int b = 0; b < no_borders; ++b) 
        buffer[b] = new int[m];

    int* temp_row = new int[m];

    for (int i = 0; i < n; ++i) {
        // Calcul rand i pe baza datelor vechi (din f sau din buffer)
        for (int j = 0; j < m; ++j) {
            int suma = 0;

            for (int ki = -no_borders; ki <= no_borders; ++ki) {
                int x = i + ki;
                int x_clamped = std::clamp(x, 0, n - 1);

                int* source;
                // Randul clamped este < i => valoarea originala = buffer
                if (x_clamped < i) {
                    int buf_idx = x_clamped % no_borders;
                    if (buf_idx < 0) buf_idx += no_borders;
                    source = buffer[buf_idx];
                } else {
                    source = &f[x_clamped * m];
                }

                for (int kj = -no_borders; kj <= no_borders; ++kj) {
                    int y = std::clamp(j + kj, 0, m - 1);
                    suma += source[y] * c[(ki + no_borders) * k + (kj + no_borders)];
                }
            }

            temp_row[j] = suma;
        }

        // Salvez randul original in buffer
        int buf_idx = i % no_borders;
        if (buf_idx < 0) buf_idx += no_borders;
        for (int j = 0; j < m; ++j) 
            buffer[buf_idx][j] = f[i * m + j];

        // Suprascriu randul curent
        for (int j = 0; j < m; ++j) 
            f[i * m + j] = temp_row[j];
    }

    delete[] temp_row;
    for (int b = 0; b < no_borders; ++b) 
        delete[] buffer[b];
    delete[] buffer;
}

void cerceteaza_secvential_inplace() {
    auto start = chrono::high_resolution_clock::now();
    calculeaza_convolutie_inplace();
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> delta = end - start;
    cout << fixed << setprecision(6) << delta.count();

    // Salveaza rezultatul
    std::ofstream out("output_sequential.txt");
    scrie_matrice_dinamica(f, n, m, out);
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Utilizare: " << argv[0] << " <n> <m> <k>\n";
        return 1;
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    k = atoi(argv[3]);

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
    citire_matrice_dinamica(f, n, m, in);
    citire_matrice_dinamica(c, k, k, in);
    in.close();

    cerceteaza_secvential_inplace();

    eliberare_memorie(f);
    eliberare_memorie(c);

    return 0;
}
