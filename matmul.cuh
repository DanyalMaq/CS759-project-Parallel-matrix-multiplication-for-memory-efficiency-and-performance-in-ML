// Author: Nic Olsen, Jason Zhou

#ifndef MATMUL_CUH
#define MATMUL_CUH

__host__ void matmul(float *A, float *B, float *C,
                                     int numARows, int numAColumns, int numBColumns);
__global__ void GPU_fill_rand_int(float* A, const int n, float min, float max);

#endif