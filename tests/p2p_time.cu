#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "../matmul.cuh"
#include "../utils.cuh"
#include "../network.cuh"
using namespace std;

__global__ void test_memory(volatile float* arr, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = 1.0;
    }
}

int main(int argc, char** argv){
    // check n is divisible by num_gpus
    /////////////////// hardcode params for testing ///////////////////
    int num_gpus = 2, n = 16384; // 2^14 * 2^14 matrix
    int th_per_block = 1024;
    int nRowsA = n, nColsA = n, nColsB = n; // test square matrices for now
    int matrix_size = num_gpus * nRowsA * nColsA; // Total size of matrix

    printf("Hardcoding params for testing\n");
    printf("n = %d, num_gpus = %d\n", n, num_gpus);

    // grid and block sizes
    dim3 threadsPerBlock(th_per_block);
    int blocks_per_dim = (matrix_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    dim3 blocksPerGrid(blocks_per_dim);
    // Set up operands and result on device 0 
    float* defaultArrA;
    float* deviceArr;
    float* hostArr = (float*)malloc(100 * sizeof(float));

    cudaEvent_t start, stop;
    float time;

    // Use managed for async memcpy
    cudaSetDevice(0);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrA, matrix_size  * sizeof(float))); 
    
    // Enable bi-directional peer access
    set_p2p_access(num_gpus);
    
    // Test access speed from peer device without prefetching
    cudaSetDevice(1);
    // Allocate memory on device 1
    CHECK_CUDA_ERROR(cudaMalloc((void**)&deviceArr, matrix_size  * sizeof(float)));
    CHECK_CUDA_ERROR(cudaEventCreate(&start)); CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    test_memory<<<blocksPerGrid, threadsPerBlock>>>(defaultArrA, matrix_size);
    kernel_err_check();

    // End of timing
    cudaEventRecord(stop);
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Time to access peer device memory without prefetching:  %3.1f ms \n", time);

    // Check device id    
    cudaPointerAttributes attr;
    CHECK_CUDA_ERROR(cudaPointerGetAttributes(&attr, defaultArrA));
    printf("Device id of arr before copy: %d\n", attr.device);
    CHECK_CUDA_ERROR(cudaPointerGetAttributes(&attr, hostArr));
    printf("Device id of host array: %d\n", attr.device);

    // Time p2p memcpy
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMemcpyPeer(defaultArrA, 0, deviceArr, 1, matrix_size * sizeof(float)));
    cudaEventRecord(stop);
    
    // Check device id
    cudaPointerGetAttributes(&attr, deviceArr);
    printf("Device id of arr after copy: %d\n", attr.device);
    // End of timing
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Time to cudaMemcpyPeer %lu bytes:  %3.1f ms \n", matrix_size * sizeof(float), time);
    
    // with prefetching
    cudaEventRecord(start);
    test_memory<<<blocksPerGrid, threadsPerBlock>>>(deviceArr, matrix_size);
    kernel_err_check();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Time to access peer device memory with prefetching:  %3.1f ms \n", time);

    // Finally,test normal copy speed instead of p2p copy
    set_p2p_access(false);
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMemcpy(defaultArrA, deviceArr, matrix_size * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to cudaMemcpy  %lu bytes:  %3.1f ms \n", matrix_size * sizeof(float), time);

}