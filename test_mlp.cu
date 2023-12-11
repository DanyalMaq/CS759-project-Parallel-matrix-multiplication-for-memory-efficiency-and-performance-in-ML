#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "matmul.cuh"
#include "utils.cuh"
#include "network.cuh"
// #include "cnpy.h"

int main()
{
    LayerDim* layer_dims = new LayerDim[3]

    // Simple test network
    layer_dims[0] = LayerDim(100, 100);
    layer_dims[1] = LayerDim(100, 100);
    layer_dims[2] = LayerDim(100, 100);

    // // For propr network
    // layer_dims[0] = LayerDim(784, 150);
    // layer_dims[1] = LayerDim(150, 200);
    // layer_dims[2] = LayerDim(200, 10);

    MLP network = MLP(3, layer_dims, 2, 100);
}