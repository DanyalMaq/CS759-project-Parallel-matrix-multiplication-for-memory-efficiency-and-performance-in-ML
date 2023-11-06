#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>

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


__host__ void matmul(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    unsigned int blocks = (n + block_dim -1)/block_dim;
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blocks , blocks);
    unsigned int sharedMem = (2 * block_dim * block_dim) * sizeof(double);
    matmul_kernel<double><<<dimGrid, dimBlock, sharedMem>>>(A, B, C, n);
    cudaDeviceSynchronize();
}