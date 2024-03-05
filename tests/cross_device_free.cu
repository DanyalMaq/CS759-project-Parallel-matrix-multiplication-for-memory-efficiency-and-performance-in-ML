#include "../src/matmul.cuh"
#include "../src/utils.cuh"
#include <iostream>

int main(){
    float* a;
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaMalloc(&a, 4*4*sizeof(float)));
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    // test error
    CHECK_CUDA_ERROR(cudaFree(a));
    cout<< a[1];
}