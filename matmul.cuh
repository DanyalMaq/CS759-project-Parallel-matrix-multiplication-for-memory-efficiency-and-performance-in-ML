// Author: Nic Olsen, Jason Zhou

# pragma once
#include <iostream>
#include <cub/cub.cuh>

using namespace std;


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
__host__ __device__ inline float softmax(const float* vals, uint32_t idx, uint32_t N);
__host__ __device__ __forceinline__ float relu(float val);

const unsigned int TILE_WIDTH = 32; // Tile size of shared memory

__host__ void matmul_rect_softmax(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB, cudaStream_t stream = nullptr);
__host__ void matmul_rect_relu(float *A, float *B, float *C,
                                     int nRowsA, int nColsA, int nColsB, cudaStream_t stream = nullptr);

__host__ void matmul(float *A, float *B, float *C,
                    int nRowsA, int nColsA, int nColsB, 
                    cudaEvent_t start = nullptr, cudaEvent_t stop = nullptr, cudaStream_t stream = nullptr);
// No timing version
__host__ void matmul(float* A, float* B, float* C,
                    int nRowsA, int nColsA, int nColsB, cudaStream_t stream = nullptr);


__host__ void matmul_relu(float* A, float* B, float* C,
                    int nRowsA, int nColsA, int nColsB, cudaStream_t stream = nullptr);

void kernel_err_check();

__global__ void matmul_rect(float *A, float *B, float *C,
                                        int nRowsA, int nColsA, int nColsB);
template <typename T>
__host__ __device__ T relu(T val);

template <uint32_t N>
__host__ __device__ inline float softmax(const float vals[N], uint32_t idx);

// Transpose a given matrix
__host__ void transpose(float *output, const float *input, int nRows, int nCols);

// Get the the specified columns of a matrix
void columns(float *output, const float *input, int rows, int columns, int start_col, int end_col);




