#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

// Dimensiuni matrice F(n, m)
int n, m;
// Dimensiune matrice convolutie C(k, k)
int k;

void initializare_fisier(const int linii, const int coloane, const int conv) {
    std::ofstream iout("input.txt");
    iout.clear();

    srand((unsigned int)time(0));

    // Scrie matricea F (n x m)
    for (int i = 0; i < linii; i++) {
        for (int j = 0; j < coloane; j++) {
            int r = rand() % 25464;
            iout << r << ' ';
        }
        iout << '\n';
    }

    // Scrie matricea convolutie C (k x k)
    for (int i = 0; i < conv; i++) {
        for (int j = 0; j < conv; j++) {
            int r = rand() % 100;
            iout << r << ' ';
        }
        iout << '\n';
    }

    iout.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Utilizare: " << argv[0] << " <n> <m> <k>\n";
        return 1;
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    k = atoi(argv[3]);

    if (k != 3) {
        std::cerr << "Aceasta aplicatie este configurata pentru k=3.\n";
        return 1;
    }

    std::cout << "Initializare: n=" << n << " m=" << m << " k=" << k << "\n";

    initializare_fisier(n, m, k);

    std::cout << "Matricele generate in input.txt\n";
    return 0;
}
