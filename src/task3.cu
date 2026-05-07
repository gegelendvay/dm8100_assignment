/*******************************************************************
* CUDA-based GPU matrix-matrix multiplication
* C = A * B for square matrices of size N x N.
*
* - Allocates A, B, C on the host
* - Initializes A and B with a deterministic pattern
* - Copies A and B to the device
* - Computes C on the GPU using a naive CUDA kernel
* - Copies C back to the host
* - Measures and prints CUDA runtime
* - Prints checksum
*
* Run:
* nvcc lib/matrix.c src/task3.cu -o build/task3 -O3
* ./build/task3
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/matrix.h"

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

/* Naive CUDA matrix-matrix multiplication:
* - 2D grid of 2D blocks
* - each thread computes a single C[row, col]
* - row-major layout
*/
__global__ static void matmul_cuda(const double *A, const double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    int N = 1024; /* Default matrix size */
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    printf("CUDA matrix-matrix multiplication, N = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(double);

    /* Allocate matrices on the host */
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(C);
        return EXIT_FAILURE;
    }

    init_matrix(A, N);
    init_matrix(B, N);

    /* Allocate matrices on the device */
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_cuda<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    double cs = checksum(C, N);
    printf("Time (ms): %.3f\n", elapsed_ms);
    printf("Checksum(C): %.12e\n", cs);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free(A);
    free(B);
    free(C);

    return 0;
}
