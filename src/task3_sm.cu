/*******************************************************************
 * CUDA-based matrix-matrix multiplication on the GPU.
 * C = A * B for square matrices of size N x N.
 *
 * - Allocates A, B, C on the host
 * - Initializes A and B with a deterministic pattern via init_matrix
 * - Copies A and B to the device
 * - Launches a grid-stride CUDA kernel that computes C = A * B
 * - Times only the kernel execution with CUDA events (kernel time)
 * - Copies C back to the host
 * - Prints kernel runtime (in seconds and milliseconds) and checksum
 *
 * This serves as the GPU T3 baseline comparable to the CPU/OpenMP/MPI
 * implementations, which also time only the main compute kernel.
 *
 * Launch configuration for strong scaling on a single GPU:
 * - Block size: fixed at THREADS_PER_BLOCK (128)
 * - Grid size:  num_blocks = num_sms * sm_multiplier
 *   where:
 *     num_sms      = number of streaming multiprocessors on the device
 *     sm_multiplier is passed as argv[2]
 *   e.g. sm_multiplier = 1  -> one block per SM
 *        sm_multiplier = 4  -> four blocks per SM
 * - Each thread computes multiple C(i,j) elements using a grid-stride loop
 *
 * Build and run:
 *   nvcc lib/matrix.c src/task3.cu -o build/task3 -O3
 *   ./build/task3             # N = 1024, sm_multiplier = 1
 *   ./build/task3 2048 1      # N = 2048, one block per SM
 *   ./build/task3 2048 4      # N = 2048, four blocks per SM
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/matrix.h"

/* Fixed threads per block:
 *  - 128 threads is a reasonable choice to keep the SM busy
 *  - This constant is kept the same across experiments so that
 *    strong scaling is driven by the number of blocks (SMs used),
 *    not by changing the block size.
 */
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
 *
 *  - A, B, C are stored in row-major layout (N x N)
 *  - We conceptually flatten C into a 1D array of length N*N
 *  - total_threads = gridDim.x * blockDim.x
 *  - Each thread starts at its own index 'tid' in [0, N*N) and then
 *    steps through the flat C array with stride 'total_threads':
 *
 *        for (idx = tid; idx < N*N; idx += total_threads) { ... }
 *
 *  - For each flat index idx, we recover (row, col) via:
 *
 *        row = idx / N
 *        col = idx % N
 *
 *  - This allows us to:
 *      * keep block size fixed (THREADS_PER_BLOCK)
 *      * vary only the grid size (number of blocks) to change how many
 *        SMs are used, which is convenient for strong scaling experiments
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

    if (argc >= 2) {
        N = atoi(argv[1]);
    }
    if (argc >= 3) {
        sm_multiplier = atoi(argv[2]);
    }

    /* Query the number of SMs on the current device (device 0) */
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms,
                                      cudaDevAttrMultiProcessorCount, 0));

    int num_blocks = num_sms * sm_multiplier;

    printf("CUDA grid-stride matrix-matrix multiplication\n");
    printf("  N               = %d\n", N);
    printf("  SMs on device   = %d\n", num_sms);
    printf("  sm_multiplier   = %d\n", sm_multiplier);
    printf("  Grid  (blocks)  = %d\n", num_blocks);
    printf("  Block (threads) = %d\n", THREADS_PER_BLOCK);

    size_t bytes = (size_t)N * N * sizeof(double);

    /* Allocate matrices on the host */
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Host allocation failed\n");
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

    /* Copy inputs to device and zero C */
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    /* Time the kernel with CUDA events (higher resolution than wall clock).
     * This is directly comparable to:
     *   - omp_get_wtime around matmul in the serial/OpenMP codes
     *   - MPI_Wtime around matmul in the MPI code
     * i.e., we time only the compute kernel, not allocation or data movement.
     */
    cudaEvent_t start, stop;
    float  elapsed_ms = 0.0f;
    double elapsed_s  = 0.0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Launch: <<<num_blocks, THREADS_PER_BLOCK>>> */
    CUDA_CHECK(cudaEventRecord(start));
    matmul_stride<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    elapsed_s = 1e-3 * (double)elapsed_ms;

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Checksum for correctness comparison with CPU/MPI versions */
    double cs = checksum(C, N);
    printf("Time (s):  %.6f\n", elapsed_s);
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
