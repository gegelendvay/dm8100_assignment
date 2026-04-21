#include "matrix.h"

/* Initialize matrix with a deterministic pattern */
void init_matrix(double *A, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (double)((i + j) % 100) / 100.0;
        }
    }
}

/* Simple checksum: sum of all elements in C */
double checksum(const double *C, int N) {
    double s = 0.0;
    for (int i = 0; i < N * N; ++i) {
        s += C[i];
    }
    return s;
}
