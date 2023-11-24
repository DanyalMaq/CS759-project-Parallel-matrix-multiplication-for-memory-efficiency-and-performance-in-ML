#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "matmul.cuh"
#include "utils.cuh"
using namespace std;

void __global__ check_matrix(float* A, int matrix_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0){
        printf("First value input: %f\nLast value input: %f\n", A[0], A[matrix_size-1]);
    }
}

int main(int argc, char** argv){
    // check n is divisible by num_gpus
    /////////////////// hardcode params for testing ///////////////////
    int num_gpus = 3, n = 32;
    int th_per_block = 1024;
    int nRowsA = n, nColsA = n, nColsB = n; // test square matrices for now
    int matrix_size = num_gpus * nRowsA * nColsA; // Total size of matrix
    int chunk_size = matrix_size / num_gpus; // Chunk going on each GPU

    printf("Hardcoding params for testing\n");
    printf("n = %d, num_gpus = %d\n", n, num_gpus);
    printf("Chunk size: %d\n", chunk_size);

    // grid and block sizes
    dim3 threadsPerBlock(th_per_block);
    int blocks_per_dim = (chunk_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    dim3 blocksPerGrid(blocks_per_dim);
    
    // Set up operands and result on device 0 
    float* defaultArrA;
    float* defaultArrB;
    float* defaultArrC;
    // Use managed for async memcpy
    cudaSetDevice(0);
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrA, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrB, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrC, matrix_size  * sizeof(float))); 

    for (int i = 0; i < matrix_size; ++i) {
        defaultArrA[i] = 1.0f;
        defaultArrB[i] = 1.0f;
        defaultArrC[i] = 0.0f;
    }
    // enable bi-directional peer access
    set_p2p_access(num_gpus);

    // randomly init and rescale the array on GPU
    cudaDeviceSynchronize();
    // printf("First value input: %f\nLast value input: %f\n", defaultArrA[0], defaultArrA[matrix_size-1]);
    
    cudaStream_t streams[num_gpus]; // Create a stream for each GPU for overlapping
    cudaEvent_t mem_events[num_gpus]; // For malloc
    float* deviceArraysA[num_gpus - 1];
    float* deviceArraysB[num_gpus - 1];
    float* deviceArraysC[num_gpus - 1];

    // Allocate chunk on each GPU
    for (int i = 1; i < num_gpus; ++i) {
        cudaSetDevice(i);
        // Set up synchronization points
        // cudaStreamCreate(&streams[i]);        
        // async malloc for overlapping computation
        cudaMallocAsync((void**)&deviceArraysA[i - 1], chunk_size * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysB[i - 1], chunk_size * sizeof(float), 0);
        cudaMallocAsync((void**)&deviceArraysC[i - 1], chunk_size * sizeof(float), 0);
    }

    // enable access from device 0 to all others
    // TODO: malloc only one chunk on device 0; use as buffer for all others
    // cudaSetDevice(0);
    for (int i = 1; i < num_gpus; ++i) {
        int start = i * chunk_size;
        cudaSetDevice(i); // must switch to memcpy target device
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysA[i - 1], i, (defaultArrA + 0), 0, chunk_size * sizeof(float), 0));
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(deviceArraysB[i - 1], i, (defaultArrB + 0), 0, chunk_size * sizeof(float), 0));
        check_matrix<<<1, 1, 0, 0>>>(deviceArraysA[i - 1], chunk_size);
        cudaStreamSynchronize(0);
    }

}