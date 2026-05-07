/*******************************************************************
 * MPI-based distributed-memory matrix-matrix multiplication
 * C = A * B for square matrices of size N x N.
 *
 * Data distribution (1D block-row):
 * - Matrix A is block-row distributed: each rank owns N/size rows
 * - Matrix B is fully replicated on all ranks (MPI_Bcast)
 * - Each rank computes its local block of C
 * - Rank 0 gathers all C blocks (MPI_Gather) and checks correctness
 *
 * Run:
 *   mpicc lib/matrix.c src/task2.c -o build/task2 -O3
 *   mpirun -np 4 ./build/task2 1024
 *******************************************************************/

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../lib/matrix.h"

/* Local matrix multiply on each rank:
 * - A_local has rows_per_proc rows and N columns
 * - B has N rows/columns and is replicated on all ranks
 * - C_local has rows_per_proc rows
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
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = NULL;

    if (!B) {
        fprintf(stderr, "Allocation failed for B on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Rank 0 holds full A and C; others only have B */
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

    /* Broadcast B from rank 0 to all ranks */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scatter rows of A to all ranks */
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                A_local, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Synchronize before timing local computation */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Compute local block of C */
    matmul_local(A_local, B, C_local, N, rows_per_proc);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* Gather local C blocks back to full C on rank 0 */
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
