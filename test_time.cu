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
    printf("Distributed matmul with managed memory\n");
    if (argc != 5){
        printf("Usage: ./t <nRowsA> <nColsA> <nColsB> <num_gpus>\n");
        return 0;    
    }

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    int nRowsA = std::stoi(argv[1]);
    int nColsA = std::stoi(argv[2]);
    int nColsB = std::stoi(argv[3]);
    int threads_per_block = 1024;
    int num_gpus = stoi(argv[4]);

    /////////////////// Prams for testing ///////////////////
    printf("Params for testing:\n");
    printf("nRowsA=%d, nColsA=%d, nColsA=%d num_gpus=%d\n", nRowsA, nColsA, nColsB, num_gpus);
    int matrix_size_A = nRowsA * nColsA; // Total size of matrix A
    int matrix_size_B = nColsA * nColsB; // Total size of matrix B
    int matrix_size_C = nRowsA * nColsB; // Total size of matrix C
    
    // Set up operands and result on device 0 
    float* defaultArrA;
    float* defaultArrB;
    float* defaultArrC;

    // Use managed for async memcpy
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrA, matrix_size_A  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrB, matrix_size_B  * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&defaultArrC, matrix_size_C  * sizeof(float))); 
    // CHECK_CUDA_ERROR(cudaMallocManaged((void**)&hostArrayD, matrix_size  * sizeof(float))); 

    // randomly init and rescale the array on GPU. Make a separate dim for memory allocation
    dim3 threadsPerBlockAlloc(threads_per_block);

    int blocks_per_dim_alloc_A = (matrix_size_A + threadsPerBlockAlloc.x - 1) / threadsPerBlockAlloc.x;
    dim3 blocksPerGridAllocA(blocks_per_dim_alloc_A);
    int blocks_per_dim_alloc_B = (matrix_size_B + threadsPerBlockAlloc.x - 1) / threadsPerBlockAlloc.x;
    dim3 blocksPerGridAllocB(blocks_per_dim_alloc_B);
    int blocks_per_dim_alloc_C = (matrix_size_B + threadsPerBlockAlloc.x - 1) / threadsPerBlockAlloc.x;
    dim3 blocksPerGridAllocC(blocks_per_dim_alloc_C);

    GPU_fill_rand_int<<<blocksPerGridAllocA, threadsPerBlockAlloc>>>(defaultArrA, matrix_size_A, 1.0f, 1.0f);
    GPU_fill_rand_int<<<blocksPerGridAllocB, threadsPerBlockAlloc>>>(defaultArrB, matrix_size_B, 1.0f, 1.0f);
    GPU_fill_rand_int<<<blocksPerGridAllocC, threadsPerBlockAlloc>>>(defaultArrC, matrix_size_C, 0.0f, 0.0f);
    kernel_err_check();
    cudaDeviceSynchronize();
    // printf("First value input: %f\nLast value input: %f\n", defaultArrA[0], defaultArrA[matrix_size-1]);

    ////////////////////// Running on one GPU /////////////////////////
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    start_cpu = high_resolution_clock::now();
    matmul(defaultArrA, defaultArrB, defaultArrC,
        nRowsA, nColsA, nColsB,
        start, stop
        );


    end_cpu = high_resolution_clock::now();
    cudaEventSynchronize(stop);

    float ms_single;
    cudaEventElapsedTime(&ms_single, start, stop);

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    double ms_single_cpu = duration_sec.count(); 
    printf("Time taken for single GPU = %f\n", ms_single);

    ////////////////////// Splitting one matrix (the first one) /////////////////////////
    printf("---------------------------------------------\n");
    printf("Splitting one matrix (the first one)\n");
    int nRowsA_per_GPU = (nRowsA + num_gpus - 1)/num_gpus;
    int lastnRowsA = nRowsA - nRowsA_per_GPU*(num_gpus-1);
    lastnRowsA = (lastnRowsA <= 0 ? nRowsA_per_GPU : lastnRowsA);
    printf("nRowsA_per_GPU=%d, lastnRowsA=%d\n", nRowsA_per_GPU, lastnRowsA);
    // int nColsB_per_GPU = (nColsB + num_gpus - 1)/num_gpus;

    // Launch kernel on each GPU with appropriate configurations
    cudaEvent_t start_events[num_gpus];
    cudaEvent_t end_events[num_gpus];
    for(int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
    }
    
    start_cpu = high_resolution_clock::now();
    for (int i = 0; i < num_gpus; ++i) {  
        cudaSetDevice(i);
        int startA = i*nRowsA_per_GPU*nColsA;
        int startC = i*nRowsA_per_GPU*nColsB;

        if (i != num_gpus - 1) // inital (num_gpus - 1) chunks always of same size
        {
            matmul((defaultArrA+startA), defaultArrB, (defaultArrC+startC),
                nRowsA_per_GPU, nColsA, nColsB,
                start_events[i], end_events[i]
                );
        }
        else // last chunk may be of different sizes depending on inputs
        {
            matmul((defaultArrA+startA), defaultArrB, (defaultArrC+startC),
                lastnRowsA, nColsA, nColsB,
                start_events[i], end_events[i]
                );
        }
    }
    
 
    // wait for results
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    end_cpu = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    double ms_multiple_cpu = duration_sec.count(); 

    printf("\nMatrix sizes %d x %d and %d x %d\n", nRowsA, nColsA, nColsA, nColsB);
    // Get time taken by each GPU
    for(int i = 0; i < num_gpus; i++)
    {
        float time_in_ms;
        cudaEventElapsedTime(&time_in_ms, start_events[i], end_events[i]);
        printf("Elapsed time on device %d: %f ms\n", i, time_in_ms);
    }

    printf("Time taken for single GPU using CPU time = %lf\n", ms_single_cpu);
    printf("Time taken for %d GPUs using CPU time = %lf\n", num_gpus, ms_multiple_cpu);
    
    //Print the result
    // printf("\nA\n");
    // printMatrix(defaultArrA, nRowsA, nColsA);
    // printf("\nB\n");
    // printMatrix(defaultArrB, nColsA, nColsB);
    // printf("\nC\n");
    // printMatrix(defaultArrC, nRowsA, nColsB);
    printf("---------------------------------------------\n");
    printf("First value output: %f\nMiddle value output: %f\n", defaultArrC[0], defaultArrC[matrix_size_C/2-1]);
    printf("Last value output: %f\n", defaultArrC[matrix_size_C - 1]);

     

    /////////////Splitting both matrices/////////////////////////////
    // THIS PART IS BUGGY FOR NOW. 

    // GPU_fill_rand_int<<<blocksPerGridAllocC, threadsPerBlockAlloc>>>(defaultArrC, matrix_size_C, 0.0f, 0.0f);

    // int nColsB_per_GPU = (nColsB + num_gpus - 1)/num_gpus;
    // int lastnColsB = nColsB - nColsB_per_GPU*(num_gpus-1);
    // float* ColsB[num_gpus];
    // for (int i = 0; i < num_gpus; ++i)
    // {
    //     int start_col = i*nColsB_per_GPU;
    //     if (i != num_gpus - 1) // inital (num_gpus - 1) chunks always of same size
    //     {
    //         CHECK_CUDA_ERROR(cudaMallocManaged((void**)&ColsB[i], (nColsA*nColsB_per_GPU) * sizeof(float)));
    //         columns(ColsB[i], defaultArrB, nColsA, nColsB, start_col, (start_col+nColsB_per_GPU));
    //     }
    //     else // last chunk may be of different sizes depending of inputs
    //     {
    //         CHECK_CUDA_ERROR(cudaMallocManaged((void**)&ColsB[i], (nColsA*lastnColsB) * sizeof(float)));
    //         columns(ColsB[i], defaultArrB, nColsA, nColsB, start_col, (start_col+lastnColsB));
    //     }
    // }
    // cudaDeviceSynchronize(); 
    // for (int i = 0; i < num_gpus; i++)
    // {
    //     if (i != num_gpus - 1)
    //         printMatrix(ColsB[i], nColsA, nColsB_per_GPU);
    //     else
    //         printMatrix(ColsB[i], nColsA, lastnColsB);
    // }
    

    // Launch kernel on each GPU with appropriate configurations
    // for (int i = 0; i < num_gpus; ++i) {  
    //     cudaSetDevice(i);
    //     int startA = i*nRowsA_per_GPU*nColsA;
    //     int startC = i*nRowsA_per_GPU*nColsB;

    //     if (i != num_gpus - 1) // inital (num_gpus - 1) chunks always of same size
    //     {
    //         for (int j = 0; j < num_gpus; ++j)
    //         {
    //             startC += j*nColsB_per_GPU;
    //             printf("\n%d\n", startC);

    //             if (j != num_gpus - 1) // inital (num_gpus - 1) chunks always of same size
    //             {
    //                 matmul((defaultArrA+startA), ColsB[j], (defaultArrC+startC), nRowsA_per_GPU, nColsA, nColsB_per_GPU);
    //             }
    //             else // last chunk may be of different sizes depending of inputs
    //             {
    //                 matmul((defaultArrA+startA), ColsB[j], (defaultArrC+startC), nRowsA_per_GPU, nColsA, lastnColsB);
    //             }
    //         }
    //     }
    //     else // last chunk may be of different sizes depending of inputs
    //     {
    //         for (int j = 0; j < num_gpus; ++j)
    //         {
    //             startC += j*nColsB_per_GPU;
    //             printf("\n%d\n", startC);

    //             if (j != num_gpus - 1) // inital (num_gpus - 1) chunks always of same size
    //             {
    //                 matmul((defaultArrA+startA), ColsB[j], (defaultArrC+startC), lastnRowsA, nColsA, nColsB_per_GPU);
    //             }
    //             else // last chunk may be of different sizes depending of inputs
    //             {
    //                 matmul((defaultArrA+startA), ColsB[j], (defaultArrC+startC), lastnRowsA, nColsA, lastnColsB);
    //             }
    //         }
    //     }
    // }
    
 
    // wait for results
    // for (int i = 0; i < num_gpus; ++i) {
    //     cudaSetDevice(i);
    //     cudaDeviceSynchronize();
    //     kernel_err_check();
    // }
    
    //Print the result
    // printf("\nA\n");
    // printMatrix(defaultArrA, nRowsA, nColsA);
    // printf("\nB\n");
    // printMatrix(defaultArrB, nColsA, nColsB);
    // printf("\nC\n");
    // printMatrix(defaultArrC, nRowsA, nColsB);
    // printf("First value output: %f\nMiddle value output: %f\n", defaultArrC[0], defaultArrC[matrix_size/2-1]);
    // printf("Last value output: %f\n", defaultArrC[matrix_size - 1]);


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
