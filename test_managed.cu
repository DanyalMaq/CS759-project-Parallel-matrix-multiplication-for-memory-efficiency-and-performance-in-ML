// TODOS:
// 1. Consider the case where n is not divisible by numGPUs
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
        printf("Usage: ./t <matrix size> <numGPUs>\n");
        return 0;    
    }

    int n = std::stoi(argv[1]);
    // int threads_per_block = stoi(argv[2]);
    int threads_per_block = 1024;
    int numGPUs = stoi(argv[2]);
    
    // check n is divisible by numGPUs
    if (! (n % numGPUs == 0) ){
        printf("For now, only supports n divisible by numGPUs");
        return 0;    
    }
    /////////////////// hardcode params for testing ///////////////////
    printf("Hardcoding params for testing\n");
    printf("n=%d, numGPUs=%d\n", n, numGPUs);
    numGPUs = 2;
    int nRowsA = n, nColsA = n, nColsB = n; // test square matrices for now
    int matrix_size = numGPUs * nRowsA * nColsA; // Total size of matrix
    int chunk_size = matrix_size + n - 1 / numGPUs; // Chunk going on each GPU

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

    // randomly init and rescale the array on GPU
    GPU_fill_rand_int<<<blocksPerGrid, threadsPerBlock>>>(defaultArrA, matrix_size, 1.0f, 1.0f);
    GPU_fill_rand_int<<<blocksPerGrid, threadsPerBlock>>>(defaultArrB, matrix_size, 1.0f, 1.0f);
    kernel_err_check();
    cudaDeviceSynchronize();
    printf("First value input: %f\nLast value input: %f\n", defaultArrA[0], defaultArrA[matrix_size-1]);
    
    cudaStream_t streams[numGPUs]; // Create a stream for each GPU for overlapping
    cudaStreamCreate(&streams[0]); // Create default device stream
    cudaEvent_t mem_events[numGPUs - 1]; // For malloc
    float* deviceArraysA[numGPUs - 1];
    float* deviceArraysB[numGPUs - 1];
    float* deviceArraysC[numGPUs - 1];

    // Allocate chunk on each GPU
    for (int i = 1; i < numGPUs; ++i) {
        // cudaSetDevice(i);
        // cudaStreamCreate(&streams[i]);
        // cudaEventCreate(&mem_events[i - 1]);

        // cudaMallocAsync((void**)&deviceArraysA[i - 1], chunk_size * sizeof(float), 0);
        // cudaMallocAsync((void**)&deviceArraysB[i - 1], chunk_size * sizeof(float), 0);
        // cudaMallocAsync((void**)&deviceArraysC[i - 1], chunk_size * sizeof(float), 0);
        // cudaEvent

    }


    // enable access from device 0 to all others
    // TODO: malloc only one chunk on device 0; use as buffer for all others
    cudaSetDevice(0);
    for (int i = 1; i < numGPUs; ++i) {
        int start = i * chunk_size;
        CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(i, 0));
        // CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysA[i], i, (defaultArrA + start), 0, chunk_size * sizeof(float), 0));
        // CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysB[i], i, (defaultArrB + start), 0, chunk_size * sizeof(float), 0));

        // CHECK_CUDA_ERROR(cudaMemcpy(deviceArraysA[i - 1], defaultArrA, chunk_size * sizeof(float), cudaMemcpyDeviceToDevice));  
        // CHECK_CUDA_ERROR(cudaMemcpy(deviceArraysB[i - 1], defaultArrB, chunk_size * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Launch kernel on each GPU with appropriate configurations
    for (int i = 0; i < numGPUs; ++i) {  
        cudaSetDevice(i);
        int start = i * chunk_size;
        int end = start + chunk_size;
    
        if (i == 0)
        {
            matmul(defaultArrA, defaultArrB, defaultArrC, nRowsA, nColsA, nColsB);
        }
        else
        {
            // call matmul on device i for the chunk
            // unsigned int shared_mem_size = 2 * sizeof(float) * (blocks_per_dim / numGPUs) * (blocks_per_dim / numGPUs);
            // matmul(deviceArraysA[i - 1], deviceArraysB[i - 1], deviceArraysC[i - 1], nRowsA, nColsA, nColsB);
            // cudaSetDevice(0); // ensure correct copying to default device
            // CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(defaultArrC + start, 0, deviceArraysC[i - 1], i, chunk_size * sizeof(float), 0));
            // cudaMemcpy(deviceArraysA[i], defaultArrA, chunk_size * sizeof(float), cudaMemcpyHostToDevice);
            matmul((defaultArrA + start), (defaultArrB + start), (defaultArrC + start), nRowsA, nColsA, nColsB);
        }
    }
 
    // wait for results
    for (int i = 0; i < numGPUs; ++i) {
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
    for (int i = 1; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaFree(deviceArraysA[i]);
        cudaFree(deviceArraysB[i]);
        cudaFree(deviceArraysC[i]);
    }

    return 0;
}