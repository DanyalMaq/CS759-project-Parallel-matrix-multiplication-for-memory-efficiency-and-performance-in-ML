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
using std::chrono::duration_cast;

int main(int argc, char** argv) {
    printf("---------------------------------------------\n");
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
    high_resolution_clock::time_point* start_cpus = new high_resolution_clock::time_point[num_gpus];
    high_resolution_clock::time_point* end_cpus = new high_resolution_clock::time_point[num_gpus];
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
    // printf("Params for testing:\n");
    // printf("nRowsA=%d, nColsA=%d, nColsB=%d num_gpus=%d\n", nRowsA, nColsA, nColsB, num_gpus);
    int matrix_size_A = nRowsA * nColsA; // Total size of matrix A
    int matrix_size_B = nColsA * nColsB; // Total size of matrix B
    int matrix_size_C = nRowsA * nColsB; // Total size of matrix C
    
    ///////////////// Running only using manually allocated memory ////////////////
    // printf("Running only using manually allocated memory\n");
    int nRowsA_per_GPU = nRowsA / num_gpus;
    int lastnRowsA = (nRowsA % num_gpus == 0) ? nRowsA_per_GPU : nRowsA_per_GPU + nRowsA % num_gpus;
    // printf("nRowsA_per_GPU=%d, lastnRowsA=%d, nColsA=%d\n", nRowsA_per_GPU, lastnRowsA, nColsA);

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
        CHECK_CUDA_ERROR(cudaSetDevice(i));

        CHECK_CUDA_ERROR(cudaMalloc((void**)&devA[i], (matrix_size_A / num_gpus) * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devB[i], matrix_size_B * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devC[i], (matrix_size_C / num_gpus) * sizeof(float)));

        // Start timing from copying 
        // CHECK_CUDA_ERROR(cudaEventRecord(start_events[i]));
        start_cpus[i] = high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaMemcpy(devA[i], (hostA + (i*(matrix_size_A / num_gpus))), (matrix_size_A / num_gpus) * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(devB[i], hostB, matrix_size_B * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(devC[i], (hostC + (i*(matrix_size_C / num_gpus))), (matrix_size_C / num_gpus) * sizeof(float), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        matmul(devA[i], devB[i], devC[i],
            nRowsA_per_GPU, nColsA, nColsB, static_cast<cudaStream_t>(0));
        CHECK_CUDA_ERROR(cudaEventRecord(end_events[i])); // End timing
    }
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        // CHECK_CUDA_ERROR(cudaEventSynchronize(end_events[i]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(0));
    }
    // Get time taken by each GPU
    for(int i = 0; i < num_gpus; i++)
    {
        float time_in_ms;
        // CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_in_ms, start_events[i], end_events[i]));
        end_cpus[i] = high_resolution_clock::now();
        time_in_ms = duration_cast<duration<double, std::milli>>(end_cpus[i] - start_cpus[i]).count();
        printf("Elapsed time on device %d: %f ms\n", i, time_in_ms);
    }

    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        CHECK_CUDA_ERROR(cudaMemcpy(hostC + (i * (matrix_size_C / num_gpus)), devC[i], (matrix_size_C / num_gpus) * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // printf("---------------------------------------------\n");
    // printf("First value output: %f\nMiddle value output: %f\n", hostC[0], hostC[matrix_size_C/2-1]);
    // printf("Last value output: %f\n", hostC[matrix_size_C - 1]);

    ////////////////////// Running on one GPU /////////////////////////
    // printf("---------------------------------------------\n");
    // printf("Running on one GPU\n");
    for (int i = 0; i < matrix_size_C; i++)
    {
        hostC[i] = 0;
    }

    float* devAFull;
    float* devBFull;
    float* devCFull;
    cudaSetDevice(0);

    cudaMalloc((void**)&devAFull, matrix_size_A * sizeof(float));
    cudaMemcpy(devAFull, hostA, matrix_size_A * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&devBFull, matrix_size_B * sizeof(float));
    cudaMemcpy(devBFull, hostB, matrix_size_B * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&devCFull, matrix_size_C * sizeof(float));
    cudaMemcpy(devCFull, hostC, matrix_size_C * sizeof(float), cudaMemcpyHostToDevice);
    
    matmul(devAFull, devBFull, devCFull,
        nRowsA, nColsA, nColsB,
        start, stop
        );
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    cudaMemcpy(hostC, devCFull, matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);

    float ms_single;
    cudaEventElapsedTime(&ms_single, start, stop);
    printf("Time taken for single GPU = %f\n", ms_single);
    // printf("---------------------------------------------\n");
    // printf("First value output: %f\nMiddle value output: %f\n", hostC[0], hostC[matrix_size_C/2-1]);
    // printf("Last value output: %f\n", hostC[matrix_size_C - 1]);

    // Free allocated memory
    for (int i = 0; i < num_gpus; i++)
    {
        cudaFree(devA[i]);
        cudaFree(devB[i]);
        cudaFree(devC[i]);
    }

    for(int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(end_events[i]);
    }

    return 0;

}
