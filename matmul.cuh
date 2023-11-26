// Author: Nic Olsen, Jason Zhou

# pragma once
#include <iostream>
const unsigned int TILE_WIDTH = 32; // Tile size of shared memory

__host__ void matmul(float *A, float *B, float *C,
                                     int numARows, int numAColumns, int numBColumns);

void kernel_err_check();

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                        int nRowsA, int nColsA, int nColsB);
template <typename T>
__host__ __device__ T relu(T val);

template <uint32_t N>
__host__ __device__ inline float softmax(const float vals[N], uint32_t idx);

// header-only
enum class MatrixLayout {
	RowMajor = 0,
	ColumnMajor = 1,
};
static constexpr MatrixLayout RM = MatrixLayout::RowMajor;
static constexpr MatrixLayout CM = MatrixLayout::ColumnMajor;

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

