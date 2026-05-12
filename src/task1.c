/*******************************************************************
 * OpenMP-parallel matrix-matrix multiplication C = A * B
 * for square matrices of size N x N.
 *
 * - Allocates A, B, C on the host
 * - Initializes A and B with a deterministic pattern
 * - Computes C in parallel using OpenMP (and/or BLAS GEMM)
 * - Measures and prints runtime
 * - Prints checksum of C
 *
 * Build (example with OpenBLAS):
 *   gcc lib/matrix.c src/task1.c -o build/task1 -O3 -fopenmp -lopenblas
 *   ./build/task1 [N]
 *******************************************************************/

#include <omp.h>
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include "../lib/matrix.h"

/* OpenMP parallel matrix-matrix multiplication:
 *  - collapse(2) over (i, j) to expose more parallelism
 *  - schedule(runtime) to experiment with static/dynamic/guided
 *    via OMP_SCHEDULE environment variable
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

/* OpenMP blocked matrix-matrix multiplication:
 *  - tiles the i, j, k loops into BS x BS blocks to improve cache reuse
 *  - each thread works on tiles of C, accumulating contributions from tiles
 *    of A and B loaded from main memory
 *  - uses collapse(2) over (ii, jj) to expose parallelism at the tile level
 *  - uses static scheduling, which is appropriate for regular, balanced work
 */
static void matmul_omp_blocked(const double *A, const double *B, double *C, int N) {
    const int BS = 64;
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            for (int kk = 0; kk < N; kk += BS) {
                int i_max = (ii + BS < N) ? ii + BS : N;
                int j_max = (jj + BS < N) ? jj + BS : N;
                int k_max = (kk + BS < N) ? kk + BS : N;

                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        double sum = C[i * N + j];
                        for (int k = kk; k < k_max; ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

/* NEW: BLAS GEMM version using row-major layout via CBLAS.
 * Computes C = A * B for N x N matrices (double precision).
 */
static void matmul_blas(const double *A, const double *B, double *C, int N) {
    const double alpha = 1.0;
    const double beta  = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                alpha,
                A, N,
                B, N,
                beta,
                C, N);
}

int main(int argc, char **argv) {
    int N = 1024; /* Default matrix size */
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    printf("Using %d OpenMP threads\n", threads);
    printf("Matrix-matrix multiplication, N = %d\n", N);

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

    /* Choose which implementation to time:
     *   matmul_omp(A, B, C, N);
     *   matmul_omp_blocked(A, B, C, N);
     *   matmul_blas(A, B, C, N);   // BLAS GEMM
     */
    matmul_blas(A, B, C, N);  /* NEW: use BLAS by default */

    double t1 = omp_get_wtime();

    double cs = checksum(C, N);
    printf("Time (s): %.6f\n", t1 - t0);
    printf("Checksum(C): %.12e\n", cs);

    free(A);
    free(B);
    free(C);

    return 0;
}
