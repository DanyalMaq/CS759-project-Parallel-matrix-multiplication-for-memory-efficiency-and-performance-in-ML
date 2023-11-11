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

__global__ void addOneToElements(int* array, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        array[index] += 1;
    }
}

__host__ void printMatrix(float* array, int n)
{
    printf("Matrix:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", array[i*n + j]);
        }
        printf("\n");
    }
}


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
    numGPUs = 2;
    int nRowsA = n, nColsA = n, nColsB = n;
    int matrix_size = numGPUs * nRowsA * nColsA; // test square matrices for now
    int chunk_size = (matrix_size / numGPUs);

    // grid and block sizes
    dim3 threadsPerBlock(threads_per_block);
    int blocks_per_dim = (chunk_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    dim3 blocksPerGrid(blocks_per_dim);
    /////////////////// hardcode params for testing ///////////////////
    
    // Set up operands and result on device 0 
    float* hostArrayA;
    float* hostArrayB;
    float* hostArrayC;
    cudaMallocManaged((void**)&hostArrayA, matrix_size  * sizeof(float)); 
    cudaMallocManaged((void**)&hostArrayB, matrix_size  * sizeof(float)); 
    cudaMallocManaged((void**)&hostArrayC, matrix_size  * sizeof(float)); 

    // randomly init and rescale the array on GPU
    GPU_fill_rand_int<<<blocksPerGrid, threadsPerBlock>>>(hostArrayA, matrix_size, 1.0f, 1.0f);
    GPU_fill_rand_int<<<blocksPerGrid, threadsPerBlock>>>(hostArrayB, matrix_size, 1.0f, 1.0f);
    
    cudaStream_t streams[numGPUs]; // Create a stream for each GPU for overlapping
    float* deviceArraysA[numGPUs];
    float* deviceArraysB[numGPUs];
    float* deviceArraysC[numGPUs];

    // Allocate chunk on each GPU
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaMallocAsync((void**)&deviceArraysA[i], (matrix_size + n - 1) / numGPUs  * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysB[i], (matrix_size + n - 1) / numGPUs  * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysC[i], (matrix_size + n - 1) / numGPUs  * sizeof(float), 0);
    }


    // enable access from device 0 to all others
    cudaSetDevice(0);
    for (int i = 0; i < numGPUs; ++i) {
        int start = i * chunk_size;
        if (i != 0){ // funny but...self-to-self access will fail
            int canAccess = 0;
            CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(i, 0));
            cudaMemcpyPeerAsync(deviceArraysA[i], i, (hostArrayA + start), 0, chunk_size * sizeof(float), 0);
            cudaMemcpyPeerAsync(deviceArraysB[i], i, (hostArrayB + start), 0, chunk_size * sizeof(float), 0);
        }
    }

    // Launch kernel on each GPU with appropriate configurations
    for (int i = 0; i < numGPUs; ++i) {  
        cudaSetDevice(i);
        int start = i * chunk_size;
        int end = start + chunk_size;
    
        // call matmul on device i for the chunk
        // unsigned int shared_mem_size = 2 * sizeof(float) * (blocks_per_dim / numGPUs) * (blocks_per_dim / numGPUs);
        matmul(deviceArraysA[i], deviceArraysB[i], deviceArraysC[i], nRowsA, nColsA, nColsB);
        
        cudaSetDevice(0); // ensure correct copying to default device
        cudaMemcpyPeerAsync(hostArrayC + start, 0, deviceArraysC[i], i, chunk_size * sizeof(float), 0);
    }
 
    // wait for results
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    
    //Print the result
    printMatrix(hostArrayC, n);
    // printf("%f\n%f", hostArrayC[0], hostArrayC[n]);
    
    // Free allocated memory
    delete[] hostArrayA;
    delete[] hostArrayB;
    delete[] hostArrayC;
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaFree(deviceArraysA[i]);
        cudaFree(deviceArraysB[i]);
        cudaFree(deviceArraysC[i]);
    }

    return 0;
}