#include<cuda.h>
#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
using namespace std;

void kernel_err_check(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}


// Fill an array with random integers in [min, max]
__global__ void GPU_fill_rand_int(float* A, const int n, float min, float max) {
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

// You should implement Tiled Matrix Multiplication discussed in class
// Computes the matrix product C = AB by making 'one' call to 'matmul_kernel'.
// A, B, and C are row-major representations of nxn matrices in managed memory.
// Configures the kernel call using a 2D configuration with blocks of dimensions
// block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) 
{
        
    extern __shared__ unsigned char shared_mem[];
    alignas(alignof(T)) T *As = reinterpret_cast<T *>(shared_mem);    // 1d array of block_dim * block_dim
    alignas(alignof(T)) T *Bs = reinterpret_cast<T *>(shared_mem + block_dim * block_dim * sizeof(T));    // 1d array of block_dim * block_dim
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_dim * by;
    int aEnd = aBegin + n - 1;
    int aStep = block_dim;

    int bBegin = block_dim * bx;
    int bStep = block_dim * n;
    T Csub = 0; // sum

    int c = n * block_dim * by + block_dim * bx; // subblock index
    // check in bound 
    if (c + n * ty + tx >= n * n) 
        return;

    // loop over subblocks; each thread computes one element
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) 
    {
        // check if operand in bound
        if (a + n * ty + tx >= n * n || b + n * ty + tx >= n * n) 
            continue;
        // load subblock into shared memory
        As[ty * block_dim + tx] = A[a + n * ty + tx];
        Bs[ty * block_dim + tx] = B[b + n * ty + tx];

        __syncthreads();

        // compute subblock
        for (int k = 0; k < block_dim; ++k) 
        {
            Csub += As[ty * block_dim + k] * Bs[k * block_dim + tx];
        }
        __syncthreads();
    }
    
    // write subblock to global memory
    C[c + n * ty + tx] = Csub;
}


// Use template to formulate your answer
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim)
{
    // config block
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    unsigned int shared_mem_size = 2 * sizeof(int) * block_dim * block_dim;
    matmul_kernel<<< dimGrid, dimBlock, shared_mem_size >>>(A, B, C, n, block_dim);
    
    kernel_err_check();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim)
{
    // config block
    dim3 dimBlock(block_dim, block_dim);
    // round dims
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    unsigned int shared_mem_size = 2 * sizeof(float) * block_dim * block_dim;
    matmul_kernel<<< dimGrid, dimBlock, shared_mem_size >>>(A, B, C, n, block_dim);
    
    kernel_err_check();

}
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim)
{
    // config block
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    unsigned int shared_mem_size = 2 * sizeof(double) * block_dim * block_dim;
    matmul_kernel<<< dimGrid, dimBlock, shared_mem_size >>>(A, B, C, n, block_dim);
    kernel_err_check();
}

template <typename T>
__host__ void matmul_univ(const T *A, const T *B, T *C, unsigned int n,
                       unsigned int block_dim)
{
    // config block
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    unsigned int shared_mem_size = 2 * sizeof(T) * block_dim * block_dim;
    matmul_kernel<<< dimGrid, dimBlock, shared_mem_size >>>(A, B, C, n, block_dim);
    kernel_err_check();
}
