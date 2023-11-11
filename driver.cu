#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char* argv[]) {
    // int n = atoi(argv[1]);
    // int threads = atoi(argv[2]);
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int n = 1000;
    int m = 200;

    // float
    float* Afloat = new float[n*m];
    float* Bfloat = new float[m*n];;
    float* Cfloat = new float[n*n];;
    float* AfloatD;
    float* BfloatD;
    float* CfloatD;
    // cudaMallocManaged(&Afloat, (n * m) * sizeof(float));
    // cudaMallocManaged(&Bfloat, (n * m) * sizeof(float));
    // cudaMallocManaged(&Cfloat, (n * m) * sizeof(float));
    
    for (unsigned int i = 0; i < n * m; i++) {
        Afloat[i] = 1.0;
    }
    for (unsigned int i = 0; i < m * n; i++) {
        Bfloat[i] = 1.0;
    }
    for (unsigned int i = 0; i < n * n; i++) {
        Cfloat[i] = 0.0;
    }

    cudaMalloc((void**)&AfloatD, n * m * sizeof(float));
    cudaMemcpy(AfloatD, Afloat, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&BfloatD, m * n * sizeof(float));
    cudaMemcpy(BfloatD, Bfloat, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&CfloatD, n * n * sizeof(float));
    cudaMemcpy(CfloatD, Cfloat, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    matmul(AfloatD, BfloatD, CfloatD, n, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(Cfloat, CfloatD, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("%f\n%f\n%f\n", Cfloat[0], Cfloat[n * n - 1], ms);

    cudaFree(AfloatD);
    cudaFree(BfloatD);
    cudaFree(CfloatD);
}