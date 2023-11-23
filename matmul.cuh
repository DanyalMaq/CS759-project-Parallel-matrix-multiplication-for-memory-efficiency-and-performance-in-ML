// Author: Nic Olsen, Jason Zhou

#ifndef MATMUL_CUH
#define MATMUL_CUH
#include <iostream>
const unsigned int TILE_WIDTH = 32; // Tile size of shared memory

__host__ void matmul(float *A, float *B, float *C,
                                     int numARows, int numAColumns, int numBColumns);
__global__ void GPU_fill_rand_int(float* A, const int n, float min, float max);
void kernel_err_check();
__host__ void printMatrix(float* array, int n);
__global__ void addOneToElements(int* array, int n);
__host__ void set_p2p_access(int num_gpus, bool enable = true);
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                        int nRowsA, int nColsA, int nColsB);

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

#endif