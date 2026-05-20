/*******************************************************************
 * CUDA-based tiled matrix-matrix multiplication on the GPU.
 * C = A * B for square matrices of size N x N.
 *
 * - Allocates A, B, C on the host
 * - Initializes A and B with random values in [0, 1]
 * - Copies A and B to the device
 * - Launches a tiled CUDA kernel using shared memory
 * - Uses CUDA events to time only kernel execution
 * - Copies C back to host
 * - Computes checksum for correctness verification
 *
 * This version (v2) differs from v1 by:
 *   - Using shared memory tiling (TILE_WIDTH x TILE_WIDTH)
 *   - Reducing global memory accesses via reuse within blocks
 *   - Each thread computes one element of C
 *
 * Launch configuration:
 *   - Block size: TILE_WIDTH x TILE_WIDTH
 *   - Grid size:  (N / TILE_WIDTH) x (N / TILE_WIDTH), rounded up
 *
 * Build and run:
 *   nvcc src/task3Version2.cu -o build/task3Version2 -O3
 *   ./build/task3Version2
 *******************************************************************/

#include <time.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------
 * CUDA error checking macro (abort on failure)
 * ------------------------------------------------------------------ */
#define CUDA_CHECK(err) \
    { \
        if (err != cudaSuccess) { \
            printf("%s in %s at line %d \n", \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

/* ------------------------------------------------------------------
 * Tile width (shared memory block size)
 * Must be compile-time constant for shared memory arrays
 * ------------------------------------------------------------------ */
#define TILE_WIDTH 16

/* ------------------------------------------------------------------
 * CPU reference implementation (optional validation)
 * ------------------------------------------------------------------ */
void multiply_matrix_original(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

/* ------------------------------------------------------------------
 * Tiled matrix multiplication kernel (shared memory version)
 *
 * Key idea:
 *   - Each block computes a TILE_WIDTH x TILE_WIDTH sub-block of C
 *   - Tiles of A and B are loaded into shared memory (fast reuse)
 *   - Each thread accumulates partial sums over multiple phases
 *
 * Memory behavior:
 *   - Global memory loads reduced via reuse in shared memory
 *   - Synchronization required between phases
 * ------------------------------------------------------------------ */
__global__ void tiled_multiply_matrix(double* A, double* B, double* C, int N) {
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int col = TILE_WIDTH * blockIdx.x + threadIdx.x;

    /* Shared memory tiles for A and B */
    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    int num_phases = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    double value = 0.0;

    /* Iterate over all tile phases */
    for (int phase = 0; phase < num_phases; phase++)
    {
        int a_col = phase * TILE_WIDTH + threadIdx.x;
        int b_row = phase * TILE_WIDTH + threadIdx.y;

        /* Load tile of A into shared memory */
        if ((row < N) && (a_col < N))
            sh_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            sh_A[threadIdx.y][threadIdx.x] = 0.0;

        /* Load tile of B into shared memory */
        if ((b_row < N) && (col < N))
            sh_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            sh_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        /* Compute partial dot product for this tile */
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];

        __syncthreads();
    }

    /* Write final result */
    if (row < N && col < N)
        C[row * N + col] = value;
}

/* ------------------------------------------------------------------
 * Checksum for correctness verification
 * ------------------------------------------------------------------ */
double checksum(double *C, int N) {
    double sum = 0.0;
    for (int i = 0; i < N * N; i++) {
        sum += C[i];
    }
    return sum;
}

/* ------------------------------------------------------------------
 * Main program
 * ------------------------------------------------------------------ */
int main() {
    int N = 1024;

    /* Host allocations */
    double *A  = (double*)malloc(N * N * sizeof(double));
    double *B  = (double*)malloc(N * N * sizeof(double));
    double *C  = (double*)malloc(N * N * sizeof(double));
    double *CC = (double*)malloc(N * N * sizeof(double));

    srand(time(NULL));

    /* Initialize matrices with random values in [0, 1] */
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    /* Device pointers */
    double *d_A;
    double *d_B;
    double *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice));

    /* Launch configuration:
     * Each block is TILE_WIDTH x TILE_WIDTH
     */
    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);

    /* CUDA event timing (kernel only) */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    tiled_multiply_matrix<<<grid_size, block_size>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Kernel time: %f ms\n", milliseconds);
    printf("Kernel time: %f seconds\n", milliseconds / 1000.0f);

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    /* Verify result */
    double checksum_C = checksum(C, N);
    printf("Checksum: %f\n", checksum_C);

    /* Optional CPU verification (disabled)
    multiply_matrix_original(A, B, CC, N);
    double checksum_CC = checksum(CC, N);
    printf("Check checksum: %f\n", checksum_CC);
    */

    /* Cleanup */
    free(A);
    free(B);
    free(C);
    free(CC);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
