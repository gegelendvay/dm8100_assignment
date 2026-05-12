/*******************************************************************
 * CUDA-based GPU matrix-matrix multiplication
 * C = A * B for square matrices of size N x N.
 *
 * Launch configuration (strong scaling experiment):
 * - Block size: fixed at THREADS_PER_BLOCK (128) to fill one SM
 * - Grid size:  num_sms * sm_multiplier, passed as argv[2]
 *   e.g. <<<1*sm_mul, 128>>> uses 1 SM, <<<80*sm_mul, 128>>> uses all
 * - Each thread covers multiple output elements via a grid-stride loop
 *
 * - Allocates A, B, C on the host
 * - Initializes A and B with a deterministic pattern via init_matrix
 * - Copies A and B to the device
 * - Runs the grid-stride kernel and measures time with CUDA events
 * - Copies C back to the host
 * - Prints kernel time and checksum
 *
 * Run:
 *   nvcc lib/matrix.c src/task3.cu -o build/task3 -O3
 *   ./build/task3 [N] [sm_multiplier]
 *   ./build/task3 1024 1      # one SM worth of blocks
 *   ./build/task3 1024 4      # four SMs worth of blocks
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/matrix.h"

/* Fixed threads per block: 128 threads saturate one SM's warp slots */
#define THREADS_PER_BLOCK 128

/* Convenience macro: check every CUDA API call and abort on error */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* -----------------------------------------------------------------------
 * Grid-stride matrix-matrix multiplication kernel:
 * - 1D block of THREADS_PER_BLOCK (128) threads
 * - 1D grid of num_blocks = num_sms * sm_multiplier blocks
 * - Total elements: N*N; each thread handles ceil(N*N / total_threads)
 *   elements by stepping through the flat output array with stride
 *   gridDim.x * blockDim.x (the total number of active threads)
 * - Thread index is mapped to (row, col) via integer division / modulo
 * - This allows strong scaling: fix block size at 128, vary grid size
 *   to add more SMs and observe speedup
 * ----------------------------------------------------------------------- */
__global__ static void matmul_stride(const double *A, const double *B,
                                     double *C, int N) {
    int total_threads = gridDim.x * blockDim.x;
    int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems   = N * N;

    /* Each thread strides over the flat C array */
    for (int idx = tid; idx < total_elems; idx += total_threads) {
        int row = idx / N;
        int col = idx % N;

        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[idx] = sum;
    }
}

int main(int argc, char **argv) {
    int N             = 1024; /* Default matrix size */
    int sm_multiplier = 1;    /* Default: one block per SM */

    if (argc >= 2) N             = atoi(argv[1]);
    if (argc >= 3) sm_multiplier = atoi(argv[2]);

    /* Query the number of SMs on the current device */
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms,
                                     cudaDevAttrMultiProcessorCount, 0));

    int num_blocks = num_sms * sm_multiplier;

    printf("CUDA grid-stride matrix-matrix multiplication\n");
    printf("  N              = %d\n", N);
    printf("  SMs on device  = %d\n", num_sms);
    printf("  sm_multiplier  = %d\n", sm_multiplier);
    printf("  Grid  (blocks) = %d\n", num_blocks);
    printf("  Block (threads)= %d\n", THREADS_PER_BLOCK);

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
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    /* Time the kernel with CUDA events (higher resolution than wall clock) */
    cudaEvent_t start, stop;
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Launch: <<<num_sms * sm_multiplier, 128>>> */
    CUDA_CHECK(cudaEventRecord(start));
    matmul_stride<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    double cs = checksum(C, N);
    printf("Time (ms): %.3f\n", elapsed_ms);
    printf("Checksum(C): %.12e\n", cs);

    /* Clean up device resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    /* Clean up host resources */
    free(A);
    free(B);
    free(C);

    return 0;
}
