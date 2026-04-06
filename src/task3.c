#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void multiply_matrix(double *A, double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double c_sum = 0.0;
        for (int i=0; i<N; i++){
            c_sum += A[row*N + i] * B[i*N + col];
        }
        C[row*N + col] = c_sum;
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

    double *d_A 
    double *d_B
    double *d_C

    cudaMalloc((void**)&d_A, N*N*sizeof(double));
    cudaMalloc((void**)&d_B, N*N*sizeof(double));
    cudaMalloc((void**)&d_C, N*N*sizeof(double));

    cudaMemcpy(d_A, A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float) * N, cudaMemcpyHostToDevice);

    //each thread computes one of the C matrix elements 
    //change params to see which is best  
    grid_size =
    block_size = 

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    multiply_matrix<<<grid_size,block_size>>>(A, B, C, N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    cudaMemcpy(C, d_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time: %f seconds\n", time);

    double checksum_C = checksum(C, N);
    printf("Checksum: %f\n", checksum_C);

    free(A);
    free(B);
    free(C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
