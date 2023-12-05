#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <ctime>
#include <chrono>
#include "matmul.cuh"
#include "utils.cuh"
using namespace std;

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char** argv) {
    printf("Distributed matmul with manually allocated memory\n");
    if (argc != 5){
        printf("Usage: ./t <nRowsA> <nColsA> <nColsB> <num_gpus>\n");
        return 0;    
    }

    int nRowsA = std::stoi(argv[1]);
    int nColsA = std::stoi(argv[2]);
    int nColsB = std::stoi(argv[3]);
    int threads_per_block = 1024;
    int num_gpus = stoi(argv[4]);

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEvent_t start_events[num_gpus];
    cudaEvent_t end_events[num_gpus];
    for(int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
    }

    /////////////////// Prams for testing ///////////////////
    printf("Params for testing:\n");
    printf("nRowsA=%d, nColsA=%d, nColsA=%d num_gpus=%d\n", nRowsA, nColsA, nColsB, num_gpus);
    int matrix_size_A = nRowsA * nColsA; // Total size of matrix A
    int matrix_size_B = nColsA * nColsB; // Total size of matrix B
    int matrix_size_C = nRowsA * nColsB; // Total size of matrix C
    
    ///////////////// Running only using manually allocated memory ////////////////
    printf("---------------------------------------------\n");
    printf("Running only using manually allocated memory\n");
    int nRowsA_per_GPU = (nRowsA + num_gpus - 1)/num_gpus;
    int lastnRowsA = nRowsA - nRowsA_per_GPU*(num_gpus-1);
    lastnRowsA = (lastnRowsA <= 0 ? nRowsA_per_GPU : lastnRowsA);
    printf("nRowsA_per_GPU=%d, lastnRowsA=%d\n", nRowsA_per_GPU, lastnRowsA);

    float* hostA = new float[matrix_size_A];
    float* hostB = new float[matrix_size_B];
    float* hostC = new float[matrix_size_C];
    
    for (int i = 0; i < matrix_size_A; i++)
    {
        hostA[i] = 1;
    }
    for (int i = 0; i < matrix_size_B; i++)
    {
        hostB[i] = 1;
    }
    for (int i = 0; i < matrix_size_C; i++)
    {
        hostC[i] = 0;
    }

    float* devA[num_gpus];
    float* devB[num_gpus];
    float* devC[num_gpus];
    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);

        cudaMalloc((void**)&devA[i], (matrix_size_A / num_gpus) * sizeof(float));
        cudaMemcpy(devA[i], (hostA + (i*(matrix_size_A / num_gpus))), (matrix_size_A / num_gpus) * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&devB[i], matrix_size_B * sizeof(float));
        cudaMemcpy(devB[i], hostB, matrix_size_B * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&devC[i], (matrix_size_C / num_gpus) * sizeof(float));
        cudaMemcpy(devC[i], (hostA + (i*(matrix_size_C / num_gpus))), (matrix_size_C / num_gpus) * sizeof(float), cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        matmul(devA[i], devB[i], devC[i],
        nRowsA_per_GPU, nColsA, nColsB,
        start_events[i], end_events[i]
        );
    }
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    // Get time taken by each GPU
    for(int i = 0; i < num_gpus; i++)
    {
        float time_in_ms;
        cudaEventElapsedTime(&time_in_ms, start_events[i], end_events[i]);
        printf("Elapsed time on device %d: %f ms\n", i, time_in_ms);
    }

    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(hostC + (i * (matrix_size_C / num_gpus)), devC[i], (matrix_size_C / num_gpus) * sizeof(float), cudaMemcpyDeviceToHost);
    }

    printf("---------------------------------------------\n");
    printf("First value output: %f\nMiddle value output: %f\n", hostC[0], hostC[matrix_size_C/2-1]);
    printf("Last value output: %f\n", hostC[matrix_size_C - 1]);

    // Free allocated memory
    cudaFree(defaultArrA);
    cudaFree(defaultArrB);
    cudaFree(defaultArrC);

    for(int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(end_events[i]);
    }

    return 0;

}
