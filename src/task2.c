/*******************************************************************
* MPI-parallel matrix-matrix multiplication C = A * B
* for square matrices of size N x N.
*
* - Allocates A, B, C on the host
* - Initializes A and B with a deterministic pattern
* - Distributes A by block rows across MPI ranks
* - Broadcasts B to all ranks
* - Computes the local block of C on each rank
* - Gathers C back to rank 0
* - Measures and prints runtime
* - Prints checksum of C
*
* Build and run:
*   mpicc lib/matrix.c src/task2.c -o build/task2 -O3
*   mpirun -np <p> --bind-to core --map-by core ./build/task2 [N]
*******************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "../lib/matrix.h"

/* Naive local matrix multiplication:
* - Computes C_local = A_local * B
* - Uses the standard i-j-k loop order
* - Each rank computes its own block of rows
*/
static void matmul_local(const double *A_local, const double *B, double *C_local, int N, int rows_per_proc) {
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A_local[i * N + k] * B[k * N + j];
            }
            C_local[i * N + j] = sum;
        }
    }
}

/* Cache-blocked local matrix multiplication:
* - Tiles the computation into BS x BS blocks
* - Reuses A_local[i, k] across the inner j loop
* - Accesses B[k, j] and C_local[i, j] with unit stride
* - Improves cache locality for row-major storage
*/
static void matmul_local_blocked(const double *A_local, const double *B, double *C_local, int N, int rows_per_proc) {
    const int BS = 64; /* tune for target machine */

    for (int i = 0; i < rows_per_proc * N; ++i) {
        C_local[i] = 0.0;
    }

    for (int ii = 0; ii < rows_per_proc; ii += BS) {
        int i_max = (ii + BS < rows_per_proc) ? (ii + BS) : rows_per_proc;

        for (int kk = 0; kk < N; kk += BS) {
            int k_max = (kk + BS < N) ? (kk + BS) : N;

            for (int jj = 0; jj < N; jj += BS) {
                int j_max = (jj + BS < N) ? (jj + BS) : N;

                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double aik = A_local[i * N + k];
                        for (int j = jj; j < j_max; ++j) {
                            C_local[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/* Cache-blocked local matrix multiplication:
* - Uses the same blocked structure as the other local kernel
* - Reuses A_local[i, k] across the whole j loop
* - Accesses B[k, j] and C_local[i, j] with unit stride
* - Improves cache locality compared with the naive i-j-k ordering
*/
static void matmul_local_ikj_blocked(const double *A_local, const double *B, double *C_local, int N, int rows_per_proc) {
    const int BS = 64; /* tune for target machine */

    for (int i = 0; i < rows_per_proc * N; ++i) {
        C_local[i] = 0.0;
    }

    for (int ii = 0; ii < rows_per_proc; ii += BS) {
        int i_max = (ii + BS < rows_per_proc) ? (ii + BS) : rows_per_proc;

        for (int kk = 0; kk < N; kk += BS) {
            int k_max = (kk + BS < N) ? (kk + BS) : N;

            for (int jj = 0; jj < N; jj += BS) {
                int j_max = (jj + BS < N) ? (jj + BS) : N;

                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double aik = A_local[i * N + k];
                        for (int j = jj; j < j_max; ++j) {
                            C_local[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int N = 1024; /* Default global matrix size */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    if (rank == 0) {
        printf("MPI matrix-matrix multiplication, N = %d, size = %d\n", N, size);
    }

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: N (%d) must be divisible by number of processes (%d).\n",
                    N, size);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows_per_proc = N / size;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    /* Every rank stores the full B matrix because it is broadcast */
    B = (double *)malloc((size_t)N * N * sizeof(double));
    if (!B) {
        fprintf(stderr, "Allocation failed for B on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Rank 0 holds the full A and C matrices */
    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        C = (double *)malloc((size_t)N * N * sizeof(double));

        if (!A || !C) {
            fprintf(stderr, "Allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        init_matrix(A, N);
        init_matrix(B, N);
    }

    /* Local buffers on each rank */
    double *A_local = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));
    double *C_local = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));

    if (!A_local || !C_local) {
        fprintf(stderr, "Local allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Broadcast full matrix B from rank 0 to all ranks */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scatter block rows of A to all ranks */
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                A_local, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Synchronize before timing the computation */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Compute the local block of C */
    matmul_local_ikj_blocked(A_local, B, C_local, N, rows_per_proc);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* Gather local C blocks back to rank 0 */
    MPI_Gather(C_local, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double cs = checksum(C, N);
        printf("Time (s): %.6f\n", t1 - t0);
        printf("Checksum(C): %.12e\n", cs);
    }

    free(A_local);
    free(C_local);
    free(B);

    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
