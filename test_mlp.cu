#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "include/matmul.cuh"
#include "include/utils.cuh"
#include<complex>
#include "include/network.cuh"
#include "cnpy.h"

int main()
{
    MatrixDims* layer_dims = new MatrixDims[3];

    // Simple test network
    // layer_dims[0] = MatrixDims(100, 100);
    // layer_dims[1] = MatrixDims(100, 100);
    // layer_dims[2] = MatrixDims(100, 100);

    // // For propr network
    layer_dims[0] = MatrixDims(784, 150);
    layer_dims[1] = MatrixDims(150, 200);
    layer_dims[2] = MatrixDims(200, 10);

    float** mat_weights = new float*[3];
    string load_path[] = {"./data/data/linear_0.weight.npy", "./data/data/linear_1.weight.npy", "./data/data/linear_2.weight.npy"};

    for (int i = 0; i < 3; i++){
        mat_weights[i] = new float[layer_dims[i].nRows * layer_dims[i].nCols];
        cnpy::NpyArray arr = cnpy::npy_load(load_path[i]);
        mat_weights[i] = arr.data<float>();
    }
    
    uint32_t num_devices = 2;
    uint32_t num_layers = 3;
    uint32_t batch_size = 32;
    MLP network(num_layers, layer_dims, num_devices, mat_weights);
    network.enable_tp(batch_size);
}