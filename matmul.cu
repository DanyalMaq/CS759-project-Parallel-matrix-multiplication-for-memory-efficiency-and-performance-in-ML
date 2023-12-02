#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

void kernel_err_check(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}

class Matrix{
    public:
        uint32_t nrowBeginAs;
        uint32_t ncolBeginBs;
        float *data;
        MatrixLayout layout;

    Matrix(uint32_t nrowBeginAs, uint32_t ncolBeginBs, MatrixLayout layout=RM){
        this->nrowBeginAs = nrowBeginAs;
        this->ncolBeginBs = ncolBeginBs;
        this->layout = layout;
        this->data = data;
    }
    // TODO: override indexing operator
    
};

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB
                                     ) {
    // TODO: change to 1d. Why is 1d faster?                                    
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int nRowsB = nColsA;
    int rowBeginA = by * TILE_WIDTH * nColsA;
    int colBeginB = bx * TILE_WIDTH;

    float Cvalue = 0.0;

    // stride over the tiles along columns of A and rows of B
    for (int step = 0; step < nColsA; step += TILE_WIDTH) {
        
        // check if operands are in bounds
        if (! (rowBeginA < nRowsA * nColsA && colBeginB < nColsB && tx + step < nColsA && ty + step < nRowsB) )
            break;

        // load A's tiles into shared memory
        sA[ty][tx] = A[rowBeginA + tx + step];
        // load B's tiles into shared memory
        sB[ty][tx] = B[colBeginB + (ty + step) * nColsB];
        // printf("sA[%d][%d] = %.1f, sB[%d][%d] = %.1f\n", ty, tx, sA[ty][tx], ty, tx, sB[ty][tx]);
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j) {
            // if (rowBeginA == 0 && colBeginB == 0)
                // printf("sA[%d][%d]=%.1f,sB[%d][%d]=%.1f\n", ty, j, sA[ty][j], j, tx, sB[j][tx]);
            // if (tx < nColsB && ty < nRowsA)
            Cvalue += sA[ty][j] * sB[j][tx];
        }
        __syncthreads();
    }

    if (rowBeginA == (nRowsA-1) && colBeginB == (nColsB-1))
    {
        printf("GPU Last value output array C variable: %f\n", Cvalue);
    }
    if (rowBeginA < nRowsA && colBeginB < nColsB) {
        C[rowBeginA * nColsB + colBeginB] = Cvalue;
    }
    if (rowBeginA == 0 && colBeginB == 0)
    {
        printf("GPU input A[0] = %.1f, A[n - 1] = %.1f\n", A[0], A[nRowsA * nColsA - 1]);
        printf("GPU input B[0] = %.1f, B[n - 1] = %.1f\n", B[0], B[nColsA * nColsB - 1]);
        printf("GPU output C[0] = %.1f, C[n - 1] = %.1f\n", C[0], C[nRowsA * nColsB - 1]);
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


// get
__host__ void transpose(float *output, const float *input, int nrowBeginAs, int ncolBeginBs) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Use cublasSgeam to extract the specified range of ncolBeginBs
    const float alia = 1.0f;
    const float beta = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nrowBeginAs, ncolBeginBs, &alia,
                input, nrowBeginAs, &beta, nullptr, nrowBeginAs,
                output, nrowBeginAs);
    cudaDeviceSynchronize();
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
	for (uint32_t step = 0; step < N; ++step) {
		total += expf(vals[step]);
	}

	return expf(vals[idx]) / total;
}
