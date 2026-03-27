#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void multiply_matrix(double *A, double *B, double *C, int N, int rows_per_proc) {
    for(int i=0; i<rows_per_proc; i++) {
        for(int j=0; j<N; j++) {
            C[i*N + j] = 0.0;
            for(int k=0; k<N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

double checksum(double *C, int N) {
    double sum = 0.0;
    for(int i=0; i<N*N; i++) {
        sum += C[i];
    }
    return sum;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;
    int rows_per_proc = N/size;

    double *A = NULL, *B = NULL, *C = NULL, *local_A = NULL, *local_C = NULL;

    if(rank==0) {
        A = (double*)malloc(N*N*sizeof(double));
        B = (double*)malloc(N*N*sizeof(double));
        C = (double*)malloc(N*N*sizeof(double));

        srand(time(NULL));

        for(int i=0; i<N*N; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
    }else {
        B = malloc(N*N*sizeof(double));
    }

    local_A = (double*)malloc(rows_per_proc*N*sizeof(double));
    MPI_Scatter(A, rows_per_proc*N, MPI_DOUBLE, local_A, rows_per_proc*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    B = (double*)malloc(N*N*sizeof(double));
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_C = (double*)malloc(rows_per_proc*N*sizeof(double));
    multiply_matrix(local_A, B, local_C, N, rows_per_proc);

    MPI_Gather(local_C, rows_per_proc*N, MPI_DOUBLE, C, rows_per_proc*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank==0) {
        double checksum_C = checksum(C, N);
        printf("Checksum: %f\n", checksum_C);

        free(A);
        free(B);
        free(C);
    }

    free(local_A);
    free(B);
    free(local_C);

    MPI_Finalize();

    return 0;
}
