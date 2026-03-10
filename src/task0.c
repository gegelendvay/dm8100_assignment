#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiply_matrix(double *A, double *B, double *C, int N) {
    for(int i=0; i<N; i++) {
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

int main() {
    int N = 1024;
    double *A = (double*)malloc(N*N*sizeof(double));
    double *B = (double*)malloc(N*N*sizeof(double));
    double *C = (double*)malloc(N*N*sizeof(double));

    srand(time(NULL));

    for(int i=0; i<N*N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    multiply_matrix(A, B, C, N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time: %f seconds\n", time);

    double checksum_C = checksum(C, N);
    printf("Checksum: %f\n", checksum_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
