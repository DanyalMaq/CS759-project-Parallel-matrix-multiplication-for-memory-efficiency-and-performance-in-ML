#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <curand.h>
#include <curand_kernel.h>

void kernel_err_check(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}


const unsigned int TILE_WIDTH = 32; // Tile size of shared memory

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

    if (Row == (nRowsA-1) && Col == (nColsB-1))
    {
        printf("GPU Last value output array C variable: %f\n", Cvalue);
    }
    if (Row < nRowsA && Col < nColsB) {
        C[Row * nColsB + Col] = Cvalue;
    }
    if (Row == 0 && Col == 0)
    {
        printf("GPU First value input array A: %f\n", A[0]);
        printf("GPU First value output array C: %f\n", C[0]);
    }
}


__host__ void matmul(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB)
{
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((nColsB / TILE_WIDTH) + 1, (nRowsA / TILE_WIDTH) + 1, 1);

    matrixMultiplyShared<<<dimGrid, dimBlock>>>(A, B, C, nRowsA, nColsA, nColsB);
    kernel_err_check();
    cudaDeviceSynchronize();
}

// Fill an array with random integers in [min, max]
__global__ void  GPU_fill_rand_int(float* A, const int n, float min, float max) {
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



////////////////// helper functions //////////////////////
__global__ void addOneToElements(int* array, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        array[index] += 1;
    }
}

__host__ void printMatrix(float* array, int n)
{
    printf("Matrix:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", array[i*n + j]);
        }
       printf("\n");
    }
}
