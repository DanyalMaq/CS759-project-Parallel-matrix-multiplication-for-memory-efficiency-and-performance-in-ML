#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
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
    printf("Distributed matmul with async overlapping\n");
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
    // Use managed for async memcpy
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrA, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrB, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrC, matrix_size  * sizeof(float))); 

    // randomly init and rescale the array on GPU
    // GPU_fill_rand_int<<<blocksPerGrid, threadsPerBlock>>>(defaultArrA, matrix_size, 1.0f, 1.0f);
    // GPU_fill_rand_int<<<blocksPerGrid, threadsPerBlock>>>(defaultArrB, matrix_size, 1.0f, 1.0f);
    cudaDeviceSynchronize();
    // printf("First value input: %f\nLast value input: %f\n", defaultArrA[0], defaultArrA[matrix_size-1]);
    
    cudaStream_t streams[numGPUs]; // Create a stream for each GPU for overlapping
    cudaStreamCreate(&streams[0]); // Create default device stream
    cudaEvent_t mem_events[numGPUs - 1]; // For malloc
    float* deviceArraysA[numGPUs - 1];
    float* deviceArraysB[numGPUs - 1];
    float* deviceArraysC[numGPUs - 1];

    // Allocate chunk on each GPU
    for (int i = 1; i < numGPUs; ++i) {
        cudaSetDevice(i);
        // cudaStreamCreate(&streams[i]);
        cudaEventCreate(&mem_events[i - 1]);
        cudaMallocAsync((void**)&deviceArraysA[i - 1], chunk_size * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysB[i - 1], chunk_size * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysC[i - 1], chunk_size * sizeof(float), 0);
        cudaEventRecord(mem_events[i - 1]); // record event in the default stream
    }


    // enable access from device 0 to all others
    // TODO: malloc only one chunk on device 0; use as buffer for all others
    cudaSetDevice(0);
    for (int i = 1; i < numGPUs; ++i) {
        int start = i * chunk_size;
        CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(i, 0));
        // CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysA[i], i, (defaultArrA + start), 0, chunk_size * sizeof(float), 0));
        // CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysB[i], i, (defaultArrB + start), 0, chunk_size * sizeof(float), 0));
        cudaStreamWaitEvent(0, mem_events[i - 1], 0);
        CHECK_CUDA_ERROR(cudaMemcpy(deviceArraysA[i - 1], defaultArrA, chunk_size * sizeof(float), cudaMemcpyDeviceToDevice));  
        CHECK_CUDA_ERROR(cudaMemcpy(deviceArraysB[i - 1], defaultArrB, chunk_size * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}