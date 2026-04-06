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

void multiply_matrix_original(double *A, double *B, double *C, int N) {
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
    double *CC = (double*)malloc(N*N*sizeof(double));
    srand(time(NULL));

    for(int i=0; i<N*N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
    double *d_A;
    double *d_B;
    double *d_C;

    cudaMalloc((void**)&d_A, N*N*sizeof(double));
    cudaMalloc((void**)&d_B, N*N*sizeof(double));
    cudaMalloc((void**)&d_C, N*N*sizeof(double));

    cudaMemcpy(d_A, A, sizeof(double) * N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(double) * N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(double) * N*N, cudaMemcpyHostToDevice);

    //each thread computes one of the C matrix elements 
    //change params to see which is best  
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
               (N + block_size.y - 1) / block_size.y);


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    multiply_matrix<<<grid_size,block_size>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); 
    clock_gettime(CLOCK_MONOTONIC, &end);

    cudaMemcpy(C, d_C, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time: %f seconds\n", time);

    double checksum_C = checksum(C, N);
    printf("Checksum: %f\n", checksum_C);

    multiply_matrix_original(A, B, CC, N);
    double checksum_CC = checksum(CC, N);
    printf("Check checksum: %f\n", checksum_CC);

    free(A);
    free(B);
    free(C);
    free(CC);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
