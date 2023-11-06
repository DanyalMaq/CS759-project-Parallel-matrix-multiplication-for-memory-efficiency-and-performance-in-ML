#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>

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
__global__ void matrixMultiplyShared(double *A, double *B, double *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    __shared__ double sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
    __shared__ double sB[TILE_WIDTH][TILE_WIDTH];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    double Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++) {
        if ((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns) {
            sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j) {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns) {
        C[Row * numCColumns + Col] = Cvalue;
    }
}

__host__ void matmul(const double *A, const double *B, double *C, unsigned int numRowsA, unsigned int numColsA, unsigned int numColsB, unsigned int block_dim)
{
    unsigned int blocksX = (numColsB + block_dim - 1) / block_dim;
    unsigned int blocksY = (numRowsA + block_dim - 1) / block_dim;
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blocksX, blocksY);
    unsigned int sharedMem = (2 * block_dim * block_dim) * sizeof(double);
    matmul_kernel<double><<<dimGrid, dimBlock, sharedMem>>>(A, B, C, numRowsA, numColsA, numColsB);
    cudaDeviceSynchronize();
}

//Second attempt
__host__ void matmul2(double *A, double *B, double *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((numCColumns / TILE_WIDTH) + 1, (numCRows / TILE_WIDTH) + 1, 1);

    matrixMultiplyShared <<<dimGrid, dimBlock>>>
                                       (A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaError_t err1 = cudaPeekAtLastError();
    cudaDeviceSynchronize();
    printf("Got CUDA error ... %s \n", cudaGetErrorString(err1));
}
