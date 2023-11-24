#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

void kernel_err_check(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}

enum class MatrixLayout {
	RowMajor = 0,
	ColumnMajor = 1,
};

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
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((nColsB / TILE_WIDTH) + 1, (nRowsA / TILE_WIDTH) + 1);

    matrixMultiplyShared<<<dimGrid, dimBlock>>>(A, B, C, nRowsA, nColsA, nColsB);
    kernel_err_check();
}


///////////////////// Activations //////////////////////
template <typename T>
__host__ __device__ T relu(T val) {
	return (T)max((float)val, 0.0f);
}

// TODO: change to parallel reduction
template <uint32_t N>
__host__ __device__ inline float softmax(const float vals[N], uint32_t idx) {
	float total = 0;

	#pragma unroll
	for (uint32_t i = 0; i < N; ++i) {
		total += expf(vals[i]);
	}

	return expf(vals[idx]) / total;
}
