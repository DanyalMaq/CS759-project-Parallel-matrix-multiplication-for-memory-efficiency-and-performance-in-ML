#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void addOneToElements(int* array, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        array[index] += 1;
    }
}

int main() {
    int n = 1000;
    int* hostArray = new int[n];
    int* deviceArrays[2];
    int numGPUs = 2;
    printf("Here\n");

    // Allocate memory on each GPU for the corresponding half of the array
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaMalloc((void**)&deviceArrays[i], (n / 2) * sizeof(int));
    }

    // Initialize the array with some values
    for (int i = 0; i < n; ++i) {
        hostArray[i] = i;
    }

    // Copy first half of the array to the first GPU
    cudaSetDevice(0);
    cudaMemcpy(deviceArrays[0], hostArray, (n / 2) * sizeof(int), cudaMemcpyHostToDevice);

    // Copy second half of the array to the second GPU
    cudaSetDevice(1);
    cudaMemcpy(deviceArrays[1], hostArray + (n / 2), (n / 2) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel on each GPU with appropriate configurations
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x);
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        addOneToElements<<<blocksPerGrid, threadsPerBlock>>>(deviceArrays[i], n / 2);
    }

    // Copy data back from each GPU to host
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(hostArray + (i * (n / 2)), deviceArrays[i], (n / 2) * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Print the updated array
    std::cout << "Updated Array:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << hostArray[i] << " ";
    }
    std::cout << std::endl;

    // Free allocated memory
    delete[] hostArray;
    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);
        cudaFree(deviceArrays[i]);
    }

    return 0;
}