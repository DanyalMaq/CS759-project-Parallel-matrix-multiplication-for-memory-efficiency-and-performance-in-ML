#pragma once
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>

// Fill an array with random integers in [min, max]
__global__ void  GPU_fill_rand_int(float* A, const int n, float min, float max) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    
    // Initialize the random state for the current thread
    curandState state;
    unsigned long long seed = 759;
    curand_init(seed, idx, 0, &state);
    
    // Generate a random float and convert it to an integer
    float rnd = curand_uniform(&state); // (0.0, 1.0]
    A[idx] = static_cast<int>( rnd * (max - min) + min );
}


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

// enable bidirectional p2p access between all GPUs
__host__ void set_p2p_access(int num_gpus, bool enable = true){
    for (int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);

        for (int j = 0; j < num_gpus; j++)
            if (i != j)
                if (enable)
                    CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(j, 0));
                else
                    CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(j));
    }
}