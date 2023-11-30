// TODOS:
// 1. Consider the case where n is not divisible by num_gpus
// 2. Compare time for async and non-async
// 3. Incorporate streams within each GPU computation
// 4. Change the initial kernel to handle n x m matrix
// 5. Make it take a matrix of n x m 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "matmul.cuh"
#include "utils.cuh"
#include <string>
#include <cublas_v2.h>

void transpose(float *output, const float *input, int rows, int columns, int start_col, int end_col) {
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
    cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{
    printf("Distributed matmul with managed memory\n");
    if (argc != 3){
        printf("Usage: ./t <matrix size> <num_gpus>\n");
        return 0;    
    }

    int n = std::stoi(argv[1]);
    // int threads_per_block = stoi(argv[2]);
    int threads_per_block = 1024;
    int num_gpus = std::stoi(argv[2]);
    
    // check n is divisible by num_gpus
    if (! (n % num_gpus == 0) ){
        printf("For now, only supports n divisible by num_gpus");
        return 0;    
    }
    /////////////////// hardcode params for testing ///////////////////
    printf("Hardcoding params for testing\n");
    printf("n=%d, num_gpus=%d\n", n, num_gpus);
    num_gpus = 2;
    int nRowsA = n, nColsA = n, nColsB = n; // test square matrices for now
    int matrix_size = num_gpus * nRowsA * nColsA; // Total size of matrix
    int chunk_size = matrix_size / num_gpus; // Chunk going on each GPU

    // grid and block sizes
    dim3 threadsPerBlock(threads_per_block);
    int blocks_per_dim = (chunk_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    dim3 blocksPerGrid(blocks_per_dim);
    /////////////////// hardcode params for testing ///////////////////
    
    // Set up operands and result on device 0 
    float* defaultArrA;
    float* defaultArrB;
    float* defaultArrC;
    float* hostArrayD;
    // Use managed for async memcpy
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrA, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrB, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrC, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&hostArrayD, matrix_size  * sizeof(float))); 

    // randomly init and rescale the array on GPU. Make a separate dim for memory allocation
    dim3 threadsPerBlockAlloc(threads_per_block);
    int blocks_per_dim_alloc = (matrix_size + threadsPerBlockAlloc.x - 1) / threadsPerBlockAlloc.x;
    dim3 blocksPerGridAlloc(blocks_per_dim_alloc);
    GPU_fill_rand_int<<<blocksPerGridAlloc, threadsPerBlockAlloc>>>(defaultArrA, matrix_size, 1.0f, 2.0f);
    GPU_fill_rand_int<<<blocksPerGridAlloc, threadsPerBlockAlloc>>>(defaultArrB, matrix_size, 0.0f, 0.0f);
    kernel_err_check();
    cudaDeviceSynchronize();
    printMatrix(defaultArrA, nRowsA, nColsA);
    transpose(defaultArrB, defaultArrA, nRowsA, nColsA, 0, 8);
    printf("Printing matrix\n");
    printMatrix(defaultArrB, nColsA, nRowsA);
}