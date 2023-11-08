#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <curand.h>
#include <curand_kernel.h>

const unsigned int TILE_WIDTH = 32;

// Source(s) used for this file:
// https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
// Lecture 11 from the course
// Me 50% Stackoverflow 20% Lecture 30% 
template <typename T>
__device__ T* shared_memory_proxy()
{
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

void kernel_err_check(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Kernel error: %s\n", cudaGetErrorString(err));
}

template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int numRowsA, unsigned int numColsA, unsigned int numColsB)
{
    int xblock = blockDim.x;
    int yblock = blockDim.y;

    auto shared_memory = shared_memory_proxy<T>();
    T* As = shared_memory;
    T* Bs = As + (xblock * yblock);

    int bx = blockIdx.x; 
    int by = blockIdx.y; 

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    int aBegin = numColsA * yblock * by;
    int aEnd = aBegin + numColsA - 1; 
    int aStep = xblock; 

    int bBegin = yblock * bx;
    int bStep = yblock * numColsB;

    T Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        for (int k = 0; k < xblock; ++k) {
            int A_row_index = a / numColsA + ty;
            int A_col_index = a % numColsA + k;
            int B_row_index = (b + k) / numColsB;
            int B_col_index = (b + k) % numColsB + tx;

            As[ty * xblock + tx] = (A_row_index < numRowsA && A_col_index < numColsA) ? A[A_row_index * numColsA + A_col_index] : 0;
            Bs[ty * xblock + tx] = (B_row_index < numColsA && B_col_index < numColsB) ? B[B_row_index * numColsB + B_col_index] : 0;

            __syncthreads();

            for (int i = 0; i < xblock; ++i) {
                Csub += As[ty * xblock + i] * Bs[i * xblock + tx];
            }

            __syncthreads();
        }
    }

    int C_row_index = yblock * by + ty;
    int C_col_index = xblock * bx + tx;

    if (C_row_index < numRowsA && C_col_index < numColsB) {
        C[C_row_index * numColsB + C_col_index] = Csub;
    }
}

//Second Attempt
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB
                                     ) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int ph = 0; ph < (((nColsA - 1) / TILE_WIDTH) + 1); ph++) {
        if ((Row < nRowsA) && (threadIdx.x + (ph * TILE_WIDTH)) < nColsA) {
            sA[threadIdx.y][threadIdx.x] = A[(Row * nColsA) + threadIdx.x + (ph * TILE_WIDTH)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < nColsB && (threadIdx.y + ph * TILE_WIDTH) < nColsA) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * nColsB + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j) {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < nRowsA && Col < nColsB) {
        C[Row * nColsB + Col] = Cvalue;
    }
}

// __host__ void matmul(const float *A, const float *B, float *C, unsigned int numRowsA, unsigned int numColsA, unsigned int numColsB, unsigned int block_dim)
// {
//     unsigned int blocksX = (numColsB + block_dim - 1) / block_dim;
//     unsigned int blocksY = (numRowsA + block_dim - 1) / block_dim;
//     dim3 dimBlock(block_dim, block_dim);
//     dim3 dimGrid(blocksX, blocksY);
//     unsigned int sharedMem = (2 * block_dim * block_dim) * sizeof(float);
//     matmul_kernel<float><<<dimGrid, dimBlock, sharedMem>>>(A, B, C, numRowsA, numColsA, numColsB);
//     cudaDeviceSynchronize();
// }

//Second attempt
__host__ void matmul2(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB)
{
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((nColsB / TILE_WIDTH) + 1, (nRowsA / TILE_WIDTH) + 1, 1);

    matrixMultiplyShared <<<dimGrid, dimBlock>>>
                                       (A, B, C, nRowsA, nColsA, nColsB);

    cudaError_t err1 = cudaPeekAtLastError();
    cudaDeviceSynchronize();
    kernel_err_check();
}

// Fill an array with random integers in [min, max]
__global__ void GPU_fill_rand_int(float* A, const int n, float min, float max) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    // Initialize the random state for the current thread
    curandState state;
    unsigned long long seed = 759;
    curand_init(seed, idx, 0, &state);
    
    // Generate a random float and convert it to an integer
    float rnd = curand_uniform(&state); // (0.0, 1.0]
    A[idx] = static_cast<int>( rnd * (max - min) + min );
}