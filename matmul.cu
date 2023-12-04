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


// GPUMatrix multiplication with shared memory for non-square matrices 
__global__ void matmul_rect(float *A, float *B, float *C,
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

        // if (sB[ty][tx] != 0.0 && sA[ty][tx] != 0.0)
            // printf("sA[%d][%d] = %.1f, sB[%d][%d] = %.1f, step = %d\n", ty, tx, sA[ty][tx], ty, tx, sB[ty][tx], step);
        __syncthreads();

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
        printf("C[%d][%d] = %.1f\n", nRowsA, nColsB, C[rowBeginA * nColsB + colBeginB]);
    }
    // if (nRowsA == 0 && nColsB == 0)
    // {
    //     printf("GPU input A[0] = %.1f, A[n - 1] = %.1f\n", A[0], A[nRowsA * nColsA - 1]);
    //     printf("GPU input B[0] = %.1f, B[n - 1] = %.1f\n", B[0], B[nColsA * nColsB - 1]);
    //     printf("GPU output C[0] = %.1f, C[n - 1] = %.1f\n", C[0], C[nRowsA * nColsB - 1]);
    // }
}


__host__ void matmul(float *A, float *B, float *C,
                    int nRowsA, int nColsA, int nColsB)
{
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((nColsB / TILE_WIDTH) + 1, (nRowsA / TILE_WIDTH) + 1);
    printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y);
    matmul_rect<<<dimGrid, dimBlock>>>(A, B, C, nRowsA, nColsA, nColsB);
    kernel_err_check();
}


__host__ void transpose(float *output, const float *input, int nRows, int nCols) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Use cublasSgeam to extract the specified range of nCols
    const float a = 1.0f;
    const float b = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nRows, nCols, &a,
                input, nRows, &b, nullptr, nRows,
                output, nRows);
    // no need to sync because it uses default stream
    // cudaStreamSynchronize(0);
}

class GPUMatrix{
    public:
        uint32_t nRows;
        uint32_t nCols;
        float *data;
        MatrixLayout layout;

    GPUMatrix(uint32_t nRows, uint32_t nCols, float* data, MatrixLayout layout=RM){
        this->nRows = nRows;
        this->nCols = nCols;
        this->data = data;
        this->layout = layout;
    }

    GPUMatrix(uint32_t nRows, uint32_t nCols, MatrixLayout layout=RM, cudaStream_t stream=nullptr){
        this->nRows = nRows;
        this->nCols = nCols;
        this->layout = layout;
        cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream);
    }

    ~GPUMatrix(){
        cudaFree(data);
    }   
    
    // override indexing operatnRows
    float& operator()(uint32_t i, uint32_t j){
        if (i >= nRows || j >= nCols)
            throw std::out_of_range("Index out of range");
        return data[i * nCols + j];
    }

    const float& operator()(uint32_t i, uint32_t j) const{
        if (i >= nRows || j >= nCols)
            throw std::out_of_range("Index out of range");
        return data[i * nCols + j];
    }
    
    void T(){
        transpose(data, data, nRows, nCols);
        // swap nRows and nCols and reverse layout
        std::swap(nRows, nCols);
        layout = (layout == RM) ? CM : RM; 
    }
        
};

///////////////////// Activations //////////////////////
template <typename T>
__host__ __device__ T relu(T val) {
	return (T)max((float)val, 0.0f);
}


// TODO: change to parallel reduction
template <uint32_t N>
__host__ __device__ inline float softmax(const float vals[N], uint32_t idx) {
	float total = 0;

	// #pragma unroll
	for (uint32_t step = 0; step < N; ++step) {
		total += expf(vals[step]);
	}

	return expf(vals[idx]) / total;
}
