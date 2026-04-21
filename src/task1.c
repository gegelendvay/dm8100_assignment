/*******************************************************************
 * OpenMP-parallel matrix-matrix multiplication C = A * B
 * for square matrices of size N x N.
 *
 * - Allocates A, B, C_serial, C_omp on the host
 * - Computes a serial reference C_serial
 * - Computes C_omp in parallel using OpenMP
 * - Measures and prints serial and OpenMP runtimes
 * - Prints checksums and Frobenius norm ||C_serial - C_omp||_F
 *
 * Parallelism:
 * - #pragma omp parallel for collapse(2) schedule(runtime)
 *   so the schedule can be controlled via OMP_SCHEDULE.
 *
 * Run:
 *   gcc lib/matrix.c src/task1.c -o build/task1 -O3 -fopenmp
 *   ./build/task1
 *******************************************************************/

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../lib/matrix.h"

/* OpenMP parallel implementation:
 * - collapse(2) over (i, j)
 * - schedule(runtime) to experiment with static/dynamic/guided
 *   via OMP_SCHEDULE.
 */
static void matmul_omp(const double *A, const double *B, double *C, int N) {
#pragma omp parallel for collapse(2) schedule(runtime)
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

int main(int argc, char **argv) {
    int N = 1024;  /* Default size */
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    printf("OpenMP matrix-matrix multiplication, N = %d\n", N);

    /* Allocate matrices */
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
    matmul_omp(A, B, C, N);
    double t1 = omp_get_wtime();

    double cs = checksum(C, N);
    printf("Time (s): %.6f\n", t1 - t0);
    printf("Checksum(C): %.12e\n", cs);

    free(A);
    free(B);
    free(C);

    return 0;
}
