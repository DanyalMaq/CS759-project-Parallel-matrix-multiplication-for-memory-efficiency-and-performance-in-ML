#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "../matmul.cuh"
#include "../utils.cuh"
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
    int num_gpus = 2, n = 1024;
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float time;

    // Use managed for async memcpy
    cudaSetDevice(0);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&defaultArrA, matrix_size  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMalloc((void**)&deviceArr, matrix_size  * sizeof(float)));
    
    // Enable bi-directional peer access
    set_p2p_access(num_gpus);
    cudaDeviceSynchronize();
    
    // Test access speed from peer device without prefetching
    cudaSetDevice(1);
    cudaEventRecord(start);
    test_memory<<<blocksPerGrid, threadsPerBlock>>>(defaultArrA, matrix_size);
    kernel_err_check();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to access peer device memory without prefetching:  %3.1f ms \n", time);

    // Time p2p memcpy
    cudaEventRecord(start);
    cudaPointerAttributes attr;
    CHECK_CUDA_ERROR(cudaPointerGetAttributes(&attr, defaultArrA));
    printf("Device id before copy: %d\n", attr.device);

    cudaMemcpyPeer(defaultArrA, 0, deviceArr, 1, matrix_size * sizeof(float));

    cudaPointerGetAttributes(&attr, defaultArrA);
    printf("Device id after copy: %d\n", attr.device);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Time to copy %lu bytes from device 1 to device 0:  %3.1f ms \n", matrix_size * sizeof(float), time);
    
    // with prefetching
    cudaEventRecord(start);
    test_memory<<<blocksPerGrid, threadsPerBlock>>>(deviceArr, matrix_size);
    kernel_err_check();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Time to access peer device memory with prefetching:  %3.1f ms \n", time);

    // Finally,test normal copy speed instead of p2p copy
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMemcpy(defaultArrA, deviceArr, matrix_size * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to copy %lu bytes from device 1 to device 0:  %3.1f ms \n", matrix_size * sizeof(float), time);

}