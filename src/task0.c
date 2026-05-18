/*******************************************************************
 * Serial implementation of matrix-matrix multiplication
 * C = A * B for square matrices of size N x N.
 *
 * - Allocates A, B, C on the host
 * - Initializes A and B with a deterministic pattern
 * - Computes C = A * B
 * - Times the computation with omp_get_wtime
 * - Prints runtime and a checksum of C
 *
 * This serves as the serial T1 baseline for the other tasks.
 *
 * Build and run:
 *   gcc lib/matrix.c src/task0.c -o build/task0 -O3 -fopenmp
 *   ./build/task0
 *******************************************************************/

#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../lib/matrix.h"

/* Naive serial matrix-matrix multiplication: C = A * B (IJK order) */
static void matmul_serial(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* Naive serial matrix-matrix multiplication: C = A * B (IKJ order) */
static void matmul_serial_ikj(const double *A, const double *B, double *C, int N) {
    memset(C, 0, (size_t)N * N * sizeof(double));
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            const double a = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    int N = 1024; /* Default matrix size */
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    printf("Serial matrix-matrix multiplication, N = %d\n", N);

    /* Allocate matrices on the heap */
    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)malloc((size_t)N * N * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(C);
        return EXIT_FAILURE;
    }

    init_matrix(A, N);
    init_matrix(B, N);

    double t0 = omp_get_wtime();
    matmul_serial_ikj(A, B, C, N);
    double t1 = omp_get_wtime();

    double cs = checksum(C, N);
    printf("Time (s): %.6f\n", t1 - t0);
    printf("Checksum(C): %.12e\n", cs);

    free(A);
    free(B);
    free(C);

    return 0;
}
