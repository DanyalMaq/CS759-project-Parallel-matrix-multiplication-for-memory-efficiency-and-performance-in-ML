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
#include <string>
#include "matmul.cuh"
#include "utils.cuh"
using namespace std;

int main(int argc, char** argv){
    printf("Distributed matmul with async overlapping\n");
    if (argc != 3){
        printf("Usage: %s <matrix size> <num_gpus>\n", argv[0]);
        return 0;    
    }

    int n = std::stoi(argv[1]);
    // int th_per_block = stoi(argv[2]);
    int th_per_block = 1024;
    int num_gpus = stoi(argv[2]);
    
    if (! (n % num_gpus == 0) ){
        printf("For now, only supports n divisible by num_gpus");
        return 0;    
    }

    /////////////////// hardcode params for testing ///////////////////
    printf("Hardcoding params for testing\n");
    printf("n = %d, num_gpus = %d\n", n, num_gpus);

    num_gpus = 2;
    int nRowsA = n, nColsA = n, nColsB = n; // test square matrices for now
    int matrix_size = num_gpus * nRowsA * nColsA; // Total size of matrix
    int chunk_size = matrix_size / num_gpus; // Chunk going on each GPU
    
    printf("Chunk size: %d\n", chunk_size);

    // Set up operands and result buffers on device 0 
    float* defaultArrA;
    float* defaultArrB;
    float* defaultArrC;

    // Use managed for async memcpy
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrA, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrB, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrC, matrix_size  * sizeof(float))); 
    // enable bidirectional peer access
    set_p2p_access(num_gpus);

    // randomly init and rescale the array on GPU
    dim3 initDimGrid((matrix_size + th_per_block - 1) / th_per_block);
    GPU_fill_rand_int<<<initDimGrid, th_per_block>>>(defaultArrA, matrix_size, 1.0f, 1.0f);
    GPU_fill_rand_int<<<initDimGrid, th_per_block>>>(defaultArrB, matrix_size, 1.0f, 1.0f);
    cudaStreamSynchronize(0);
    
    // Create a stream for each GPU for pipelining
    cudaStream_t streams[num_gpus]; cudaStreamCreate(&streams[0]);
    cudaEvent_t mem_events[num_gpus]; // For malloc
    float* deviceArraysA[num_gpus - 1];
    float* deviceArraysB[num_gpus - 1];
    float* deviceArraysC[num_gpus - 1];

    // Allocate chunk on each GPU
    for (int i = 1; i < num_gpus; ++i) {
        cudaSetDevice(i);
        // Set up synchronization points (use default stream for now)
        // cudaStreamCreate(&streams[i]);
        cudaEventCreate(&mem_events[i]);
        
        // async malloc for overlapping computation
        cudaMallocAsync((void**)&deviceArraysA[i - 1], chunk_size * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysB[i - 1], chunk_size * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysC[i - 1], chunk_size * sizeof(float), 0);
        // sync barrier
        cudaEventRecord(mem_events[i], 0); 
    }

    // enable access from device 0 to all others
    // TODO: malloc only one chunk on device 0; use as buffer for all others
    for (int i = 1; i < num_gpus; ++i) {
        int start = i * chunk_size;
        cudaSetDevice(i); // must switch to memcpy target device
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysA[i - 1], i, (defaultArrA + start), 0, chunk_size * sizeof(float), 0));
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysB[i - 1], i, (defaultArrB + start), 0, chunk_size * sizeof(float), 0));
    }

    // Launch kernel on each GPU with appropriate configurations
    dim3 matmulBlock(TILE_WIDTH, TILE_WIDTH); // tile size 
    dim3 matmulGrid((nColsB / TILE_WIDTH) + 1, (nRowsA / TILE_WIDTH) + 1);
    for (int i = 0; i < num_gpus; ++i) {  
        cudaSetDevice(i);
        int start = i * chunk_size;
        if (i == 0)
            matrixMultiplyShared<<<matmulBlock, matmulGrid, 0, 0>>>(
                defaultArrA, defaultArrB, defaultArrC, nRowsA, nColsA, nColsB
            );
        else
            matrixMultiplyShared<<<matmulBlock, matmulGrid, 0, 0>>>(
                deviceArraysA[i - 1], deviceArraysB[i - 1], deviceArraysC[i - 1], nRowsA, nColsA, nColsB
            );
        
        // copy chunk back to device 0
        if (i != 0)
            CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(defaultArrC + start, 0, deviceArraysC[i - 1], i, chunk_size * sizeof(float), 0));
    }
 
    // wait for results
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(0);
    }
    
    //Print the result
    // printMatrix(defaultArrC, n);
    printf("Matmul First value output: %f\nLast value output: %f\n", defaultArrC[0], defaultArrC[matrix_size-1]);
    
    // Clean up
    set_p2p_access(num_gpus, false);
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