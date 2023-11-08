// Author: Nic Olsen, Jason Zhou

#ifndef MATMUL_CUH
#define MATMUL_CUH

// You should implement Tiled Matrix Multiplication discussed in class
// Computes the matrix product C = AB by making 'one' call to 'matmul_kernel'.
// A, B, and C are row-major representations of nxn matrices in managed memory.
// Configures the kernel call using a 2D configuration with blocks of dimensions
// block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.

// Use template to formulate your answer
__host__ void matmul(const float *A, const float *B, float *C, 
    unsigned int numRowsA, unsigned int numColsA, unsigned int numColsB, 
        unsigned int block_dim);


__host__ void matmul2(float *A, float *B, float *C,
                                     int numARows, int numAColumns, int numBColumns);
__global__ void GPU_fill_rand_int(float* A, const int n, float min, float max);

#endif