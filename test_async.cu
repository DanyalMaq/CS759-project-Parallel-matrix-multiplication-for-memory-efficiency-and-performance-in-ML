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
// #include "network.cuh"
// #include "cnpy.h"
using namespace std;

int main(int argc, char** argv){
    printf("----------------------------------------\n");
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

    int nRowsA = n, nColsA = n, nColsB = n; // test square matrices for now
    int matrix_size = num_gpus * nRowsA * nColsA; // Total size of matrix
    int chunk_size = matrix_size / num_gpus; // Chunk going on each GPU
    int def_device = 0; // default device
    
    cudaStream_t def_stream = nullptr; // device-wide default stream
    // printf("n = %d, num_gpus = %d\n", n, num_gpus);
    // printf("Per-device chunk size: %d * %d = %d\n", nRowsA, nColsA, chunk_size);

    // Set up operands and result buffers on device 0 
    float* defaultArrA;
    float* defaultArrB;
    float* defaultArrC;

    // Use managed for async memcpy
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrA, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrB, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrC, matrix_size  * sizeof(float))); 
    // enable bidirectional peer access
    set_p2p_access(num_gpus);

    // randomly init and rescale the array on GPU
    dim3 initDimGrid((matrix_size + th_per_block - 1) / th_per_block);
    GPU_fill_rand<<<initDimGrid, th_per_block>>>(defaultArrA, matrix_size, 1.0f, 1.0f);
    GPU_fill_rand<<<initDimGrid, th_per_block>>>(defaultArrB, matrix_size, 1.0f, 1.0f);
    GPU_fill_rand<<<initDimGrid, th_per_block>>>(defaultArrC, matrix_size, 0.0f, 0.0f);
    // cudaStreamSynchronize(0);
    
    // Create a stream for each GPU for pipelining
    // cudaStream_t streams[num_gpus]; cudaStreamCreate(&streams[0]);
    cudaEvent_t start_events[num_gpus]; 
    cudaEvent_t end_events[num_gpus];
    float* deviceArraysA[num_gpus - 1];
    float* deviceArraysB[num_gpus - 1];
    float* deviceArraysC[num_gpus - 1];

    // Allocate chunk on each GPU
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        
        // create events for timing
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
        // alloc chunks on each GPU
        if (i != 0){
            // async malloc for overlapping computation
            CHECK_CUDA_ERROR(cudaMallocAsync((void**)&deviceArraysA[i - 1], chunk_size * sizeof(float), def_stream));
            CHECK_CUDA_ERROR(cudaMallocAsync((void**)&deviceArraysB[i - 1], chunk_size * sizeof(float), def_stream));
            CHECK_CUDA_ERROR(cudaMallocAsync((void**)&deviceArraysC[i - 1], chunk_size * sizeof(float), def_stream));
        }
 
    }

    // Copy chunks to each GPU
    for (int i = 1; i < num_gpus; ++i) {
        int start = i * chunk_size;
        cudaSetDevice(i);
        // CHECK_CUDA_ERROR(cudaEventRecord(start_events[i], def_stream)); // Time both the kernel and memcpy
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysA[i - 1], i, (defaultArrA + start), def_device, chunk_size * sizeof(float), def_stream));
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysB[i - 1], i, (defaultArrB + start), def_device, chunk_size * sizeof(float), def_stream));
        // cudaStreamSynchronize(def_stream);
    }

    // Launch kernel on each GPU 
    for (int i = 0; i < num_gpus; ++i) {  
        cudaSetDevice(i);
        int start = i * chunk_size;

        if (i == 0){
            matmul(defaultArrA, defaultArrB, defaultArrC,
                    nRowsA, nColsA, nColsB, start_events[i], end_events[i], def_stream);
        }
        else{
            matmul(deviceArraysA[i - 1], deviceArraysB[i - 1], deviceArraysC[i - 1],
                    nRowsA, nColsA, nColsB,  start_events[i], end_events[i], def_stream);
            // all-gather to default device
            CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(defaultArrC + start, def_device, deviceArraysC[i - 1], i, chunk_size * sizeof(float), def_stream));
        }
        // CHECK_CUDA_ERROR(cudaEventRecord(end_events[i], 0));    
    }
 
    // wait for results
    float time = 0.0f;
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(def_stream);
        cudaEventElapsedTime(&time, start_events[i], end_events[i]);
        printf("Elasped time on GPU %d: %f ms\n", i, time);
    }
    
    // copy back to host
    float *results = (float*)malloc(matrix_size * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(results, defaultArrC, matrix_size * sizeof(float), cudaMemcpyDeviceToHost));
    // printMatrix(results, num_gpus * nRowsA, nColsB);
    printf("Matmul First value output: %f\n Middle value output: %f\n Last value output: %f\n",
        results[0], results[num_gpus * nRowsA * nColsB / 2 + 1], results[num_gpus * nRowsA * nColsB - 1]
    );
    

    // Clean up
    set_p2p_access(num_gpus, false);
    cudaFree(defaultArrA);
    cudaFree(defaultArrB);
    cudaFree(defaultArrC);
    for (int i = 0; i < num_gpus - 1; ++i) {
        cudaSetDevice(i);
        cudaFree(deviceArraysA[i]);
        cudaFree(deviceArraysB[i]);
        cudaFree(deviceArraysC[i]);
    }

    return 0;
}