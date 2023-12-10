#pragma once
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>


// Fill an array with random integers in [min, max]
__global__ void  GPU_fill_rand(float* A, const int n, float min, float max) {
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

__host__ void printMatrix(float* array, int n, int m)
{
    printf("\nMatrix:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%.0f ", array[i*m + j]);
        }
       printf("\n");
    }
}

// enable bidirectional p2p access between all GPUs
__host__ void set_p2p_access(int num_gpus, bool enable = true){
    // Get current device
    int current_device;
    cudaGetDevice(&current_device);
    for (int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);

        for (int j = 0; j < num_gpus; j++){

            if (i != j){
                
                // Check if accessible
                int canAccess;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                if (!canAccess) {
                    printf("WARNING: Device %d can't access peer %d\n", i, j);
                    continue;
                }
                // Set access
                if (enable)
                    cudaDeviceEnablePeerAccess(j, 0);
                else
                    cudaDeviceDisablePeerAccess(j);
            }
        }
    }
    cudaSetDevice(current_device);
}

__host__ float matmul_TFLOPS(int nRowsA, int nColsA, int nColsB, float time){
    return (2.0 * nRowsA * nColsA * nColsB) / (time * 1e12);
}