#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}
#define TILE_WIDTH 16 //needed to be a compile time constant

__global__ void tiled_multiply_matrix(double* A, double* B, double* C, int N)
{
    

    int row = TILE_WIDTH*blockIdx.y + threadIdx.y;
    int col = TILE_WIDTH*blockIdx.x + threadIdx.x;

    //shared memory  
    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    int num_phases = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    double value = 0;
    for (int phase = 0; phase < num_phases; phase++)
    {
        int a_col = phase * TILE_WIDTH + threadIdx.x;
        int b_row = phase * TILE_WIDTH + threadIdx.y;
        // Load Tiles into shared memory
        if ((row < N) && ((a_col) < N))
          sh_A[threadIdx.y][threadIdx.x] = A[(row)*N + a_col];
        else
          sh_A[threadIdx.y][threadIdx.x] = 0.0;

        if ((b_row < N) && (col < N))
          sh_B[threadIdx.y][threadIdx.x] = B[b_row*N+col];
        else
          sh_B[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N)
      C[row*N+col] = value;
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

    cudaError_t all_A = cudaMalloc((void**)&d_A, N*N*sizeof(double));
    CUDA_CHECK(all_A);
    cudaError_t all_B = cudaMalloc((void**)&d_B, N*N*sizeof(double));
    CUDA_CHECK(all_B);
    cudaError_t all_C = cudaMalloc((void**)&d_C, N*N*sizeof(double));
    CUDA_CHECK(all_C);

    cudaError_t mem_A = cudaMemcpy(d_A, A, sizeof(double) * N*N, cudaMemcpyHostToDevice);
    CUDA_CHECK(mem_A);
    cudaError_t mem_B = cudaMemcpy(d_B, B, sizeof(double) * N*N, cudaMemcpyHostToDevice);
    CUDA_CHECK(mem_B);
    cudaError_t mem_C = cudaMemcpy(d_C, C, sizeof(double) * N*N, cudaMemcpyHostToDevice);
    CUDA_CHECK(mem_C);

    //each thread computes one of the C matrix elements 
    //change params to see which is best  
    //the dimension of the blocks needs to be the same as TILE_WIDTH
    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
               (N + block_size.y - 1) / block_size.y);


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tiled_multiply_matrix<<<grid_size,block_size>>>(d_A, d_B, d_C, N);
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
