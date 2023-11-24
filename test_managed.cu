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


int main(int argc, char** argv){
    printf("Distributed matmul with managed memory\n");
    if (argc != 3){
        printf("Usage: ./t <matrix size> <num_gpus>\n");
        return 0;    
    }

    int n = std::stoi(argv[1]);
    // int threads_per_block = stoi(argv[2]);
    int threads_per_block = 1024;
    int num_gpus = stoi(argv[2]);
    
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
    GPU_fill_rand_int<<<blocksPerGridAlloc, threadsPerBlockAlloc>>>(defaultArrA, matrix_size, 1.0f, 1.0f);
    GPU_fill_rand_int<<<blocksPerGridAlloc, threadsPerBlockAlloc>>>(defaultArrB, matrix_size, 1.0f, 1.0f);
    kernel_err_check();
    cudaDeviceSynchronize();
    printf("First value input: %f\nLast value input: %f\n", defaultArrA[0], defaultArrA[matrix_size-1]);
    
    cudaStream_t streams[num_gpus]; // Create a stream for each GPU for overlapping
    cudaStreamCreate(&streams[0]); // Create default device stream
    cudaEvent_t mem_events[num_gpus - 1]; // For malloc
    float* deviceArraysA[num_gpus - 1];
    float* deviceArraysB[num_gpus - 1];
    float* deviceArraysC[num_gpus - 1];


    // Launch kernel on each GPU with appropriate configurations
    for (int i = 0; i < num_gpus; ++i) {  
        cudaSetDevice(i);
        int start = i * chunk_size;
        int end = start + chunk_size;
    
        if (i == 0)
        {
            matmul(defaultArrA, defaultArrB, defaultArrC, nRowsA, nColsA, nColsB);
        }
        else
        {
            matmul((defaultArrA+start), (defaultArrB+start), (defaultArrC+start), nRowsA, nColsA, nColsB);
        }
    }
 
    // wait for results
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    
    //Print the result
    // printMatrix(defaultArrC, n);
    printf("First value output: %f\nMiddle value output: %f\n", defaultArrC[0], defaultArrC[matrix_size/2-1]);
    printf("Last value output: %f\n", defaultArrC[matrix_size - 1]);
     
    // Free allocated memory
    cudaFree(defaultArrA);
    cudaFree(defaultArrB);
    cudaFree(defaultArrC);
    for (int i = 1; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaFree(deviceArraysA[i]);
        cudaFree(deviceArraysB[i]);
        cudaFree(deviceArraysC[i]);
    }

    return 0;
}