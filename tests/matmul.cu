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

///////////////////// Activations //////////////////////
__host__ __device__ __forceinline__ float relu(float val) {
	return (float)max((float)val, 0.0f);
}

// TODO: upgrade to online softmax
__host__ __device__ __forceinline__ float softmax(const float* vals, uint32_t idx, uint32_t N) {
    
	float total = 0;
	for (uint32_t i = 0; i < N; ++i) {
		total += expf(vals[i]);
	}

	return expf(vals[idx]) / total;
}


// GPUMatrix multiplication with shared memory for non-square matrices 
__global__ void matmul_rect(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB
                                    ) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int nRowsB = nColsA;
    int rowBeginA = by * TILE_WIDTH + ty;
    int colBeginB = bx * TILE_WIDTH + tx;

    float Ctile = 0.0;

    // stride over the tiles along columns of A and rows of B
    for (int step = 0; step < nColsA; step += TILE_WIDTH) {
        // load A's tiles into shared memory
        if (rowBeginA < nRowsA && tx + step < nColsA)
            sA[ty][tx] = A[rowBeginA * nColsA + tx + step];
        else
            sA[ty][tx] = 0.0;
        // load B's tiles into shared memory
        if (colBeginB < nColsB && ty + step < nRowsB)
            sB[ty][tx] = B[(ty + step) * nColsB + colBeginB];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();
        // if (step == 0 && sA[ty][tx] != 0.0 && sB[ty][tx] != 0.0)
            // printf("sA[%d][%d] = %.1f, sB[%d][%d] = %.1f, step = %d\n", ty, tx, sA[ty][tx], ty, tx, sB[ty][tx], step);

        for (int j = 0; j < TILE_WIDTH; ++j) {
            Ctile += sA[ty][j] * sB[j][tx];
        }

        __syncthreads();
    }
    // if (Ctile != 0.0)
        // printf("tx = %d, ty = %d, Ctile = %f\n", tx, ty, Ctile);

    // if (nRowsA == nRowsA - 1 && nColsB == nColsB - 1)
        // printf("GPU Last value output array C variable: %f\n", Ctile);

    if (rowBeginA < nRowsA && colBeginB < nColsB) {
        C[rowBeginA * nColsB + colBeginB] = Ctile;
        // printf("C[%d][%d] = %.1f\n", nRowsA, nColsB, C[rowBeginA * nColsB + colBeginB]);
    }
    // if (nRowsA == 0 && nColsB == 0)
    // {
    //     printf("GPU input A[0] = %.1f, A[n - 1] = %.1f\n", A[0], A[nRowsA * nColsA - 1]);
    //     printf("GPU input B[0] = %.1f, B[n - 1] = %.1f\n", B[0], B[nColsA * nColsB - 1]);
    //     printf("GPU output C[0] = %.1f, C[n - 1] = %.1f\n", C[0], C[nRowsA * nColsB - 1]);
    // }
}

__global__ void matmul_rect_relu(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB
                                    ) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int nRowsB = nColsA;
    int rowBeginA = by * TILE_WIDTH + ty;
    int colBeginB = bx * TILE_WIDTH + tx;

    float Ctile = 0.0;

    // stride over the tiles along columns of A and rows of B
    for (int step = 0; step < nColsA; step += TILE_WIDTH) {
        // load A's tiles into shared memory
        if (rowBeginA < nRowsA && tx + step < nColsA)
            sA[ty][tx] = A[rowBeginA * nColsA + tx + step];
        else
            sA[ty][tx] = 0.0;
        // load B's tiles into shared memory
        if (colBeginB < nColsB && ty + step < nRowsB)
            sB[ty][tx] = B[(ty + step) * nColsB + colBeginB];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j) {
            Ctile += sA[ty][j] * sB[j][tx];
        }

        __syncthreads();
    }
    if (rowBeginA < nRowsA && colBeginB < nColsB) {
        C[rowBeginA * nColsB + colBeginB] = relu(Ctile);
        // printf("C[%d][%d] = %.1f\n", nRowsA, nColsB, C[rowBeginA * nColsB + colBeginB]);
    }
}
    
__global__ void matmul_rect_softmax(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB
                                    ) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    __shared__ float rowSum[TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int nRowsB = nColsA;
    int rowBeginA = by * TILE_WIDTH + ty;
    int colBeginB = bx * TILE_WIDTH + tx;

    float Ctile = 0.0;

    // stride over the tiles along columns of A and rows of B
    for (int step = 0; step < nColsA; step += TILE_WIDTH) {
        // load A's tiles into shared memory
        if (rowBeginA < nRowsA && tx + step < nColsA)
            sA[ty][tx] = A[rowBeginA * nColsA + tx + step];
        else
            sA[ty][tx] = 0.0;
        // load B's tiles into shared memory
        if (colBeginB < nColsB && ty + step < nRowsB)
            sB[ty][tx] = B[(ty + step) * nColsB + colBeginB];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j) {
            Ctile += sA[ty][j] * sB[j][tx];
        }

        __syncthreads();
    }
    // if (rowBeginA < nRowsA && colBeginB < nColsB) {
        // C[rowBeginA * nColsB + colBeginB] = Ctile;
    // }

    // Softmax
    __shared__ float rowVals[TILE_WIDTH];
    if (rowBeginA < nRowsA && colBeginB < nColsB) {
        rowVals[tx] = Ctile;
    }else{
        rowVals[tx] = 0.0;
    }
    __syncthreads();

    float softmax_val = softmax(rowVals, tx, nColsB);
    if (rowBeginA < nRowsA && colBeginB < nColsB) {
        C[rowBeginA * nColsB + colBeginB] = softmax_val;
    }
}


__host__ void matmul(float *A, float *B, float *C,
                    int nRowsA, int nColsA, int nColsB, 
                    cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream)
{
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((nColsB / TILE_WIDTH) + 1, (nRowsA / TILE_WIDTH) + 1);
    printf("Launching %d blocks of %d threads each\n", dimGrid.x * dimGrid.y, dimBlock.x * dimBlock.y);
    
    // cudaStreamSynchronize(stream);
    // run and time the kernel 
    cudaEventRecord(start, stream);
    matmul_rect<<<dimGrid, dimBlock, 0, stream>>>(A, B, C, nRowsA, nColsA, nColsB);
    cudaEventRecord(stop, stream);
    kernel_err_check();
}

// function overload ? No timing version
__host__ void matmul(float* A, float* B, float* C,
                    int nRowsA, int nColsA, int nColsB, cudaStream_t stream)
{
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((nColsB / TILE_WIDTH) + 1, (nRowsA / TILE_WIDTH) + 1);
    
    // cudaStreamSynchronize(stream);
    // run and time the kernel 
    matmul_rect<<<dimGrid, dimBlock>>>(A, B, C, nRowsA, nColsA, nColsB);
    kernel_err_check();
}


// Transpose a given matrix
void transpose(float *output, const float *input, int nRows, int nCols) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Use cublasSgeam to extract the specified range of nCols
    const float a = 1.0f;
    const float b = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nRows, nCols, &a,
                input, nRows, &b, nullptr, nRows,
                output, nRows);
}

// Get the the specified columns of a matrix
void columns(float *output, const float *input, int rows, int columns, int start_col, int end_col) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Calculate the number of columns to extract
    int num_cols_to_extract = end_col - start_col;

    // Use cublasSgeam to extract the specified range of columns
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, num_cols_to_extract, &alpha,
                input + start_col * rows, rows, &beta, nullptr, rows,
                output, rows);
}

