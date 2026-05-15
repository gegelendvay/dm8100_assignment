#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
//Macro that checks if CUDA operation fails
#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

//CPU matrix multiplication and checksum
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

//Function that runs on the GPU. Each CUDA thread computes one element of matrix C. 
__global__ void multiply_matrix(double *A, double *B, double *C, int N) {
    //Finds which row and column thread computes. 
    //blockIdx identifies which block, blockDim size of block and threadIdx identifies which thread within the block
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) { //So that no threads outside boundaries do anything
        double c_sum = 0.0; //Actual multiplication part. Each GPU thread computes one output cell of C. It loops through one row of A and one column of B
        for (int i=0; i<N; i++){
            c_sum += A[row*N + i] * B[i*N + col];
        }
        C[row*N + col] = c_sum;
    }
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

    //GPU memory allocation. Device memory allocation. 
    cudaError_t all_A = cudaMalloc((void**)&d_A, N*N*sizeof(double));
    CUDA_CHECK(all_A);
    cudaError_t all_B = cudaMalloc((void**)&d_B, N*N*sizeof(double));
    CUDA_CHECK(all_B);
    cudaError_t all_C = cudaMalloc((void**)&d_C, N*N*sizeof(double));
    CUDA_CHECK(all_C);

    //Copy data to the GPU
    cudaError_t mem_A = cudaMemcpy(d_A, A, sizeof(double) * N*N, cudaMemcpyHostToDevice);
    CUDA_CHECK(mem_A);
    cudaError_t mem_B = cudaMemcpy(d_B, B, sizeof(double) * N*N, cudaMemcpyHostToDevice);
    CUDA_CHECK(mem_B);

    //change params to see which is best
    //Each block has 16x16 threads
    dim3 block_size(16, 16);
    //Calculates how many blocks are needed to cover the entire matrix 
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
               (N + block_size.y - 1) / block_size.y);


    //Timing GPU execution
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    multiply_matrix<<<grid_size,block_size>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); //because kernel launch is async
    clock_gettime(CLOCK_MONOTONIC, &end);

    //Copy result into CPU mem form GPU mem
    cudaError_t mem_C = cudaMemcpy(C, d_C, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
    CUDA_CHECK(mem_C);

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

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
