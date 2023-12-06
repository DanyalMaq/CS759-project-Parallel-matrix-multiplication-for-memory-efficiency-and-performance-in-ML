// Author: Nic Olsen, Jason Zhou

# pragma once
#include <iostream>
const unsigned int TILE_WIDTH = 32; // Tile size of shared memory

__host__ void matmul(float *A, float *B, float *C,
                    int nRowsA, int nColsA, int nColsB, 
                    cudaEvent_t start = nullptr, cudaEvent_t stop = nullptr, cudaStream_t stream = nullptr);

void kernel_err_check();
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


__global__ void matmul_rect(float *A, float *B, float *C,
                                        int nRowsA, int nColsA, int nColsB);
template <typename T>
__host__ __device__ T relu(T val);

template <uint32_t N>
__host__ __device__ inline float softmax(const float vals[N], uint32_t idx);

// Transpose a given matrix
__host__ void transpose(float *output, const float *input, int nRows, int nCols);

// Get the the specified columns of a matrix
void columns(float *output, const float *input, int rows, int columns, int start_col, int end_col);

////////////////////// header-only  //////////////////////
enum class MatrixLayout {
	RowMajor = 0,
	ColumnMajor = 1,
};
static constexpr MatrixLayout RM = MatrixLayout::RowMajor;
static constexpr MatrixLayout CM = MatrixLayout::ColumnMajor;
class GPUMatrix{
    public:
        uint32_t nRows;
        uint32_t nCols;
        float *data;
        MatrixLayout layout;

    GPUMatrix(uint32_t nRows, uint32_t nCols, float* data, MatrixLayout layout=RM){
        this->nRows = nRows;
        this->nCols = nCols;
        this->data = data;
        this->layout = layout;
    }

    GPUMatrix(uint32_t nRows, uint32_t nCols, MatrixLayout layout=RM, cudaStream_t stream=nullptr){
        this->nRows = nRows;
        this->nCols = nCols;
        this->layout = layout;
        cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream);
    }

    ~GPUMatrix(){
        cudaFree(data);
    }   
    
    void T(){
        transpose(data, data, nRows, nCols);
        // swap nRows and nCols and reverse layout
        std::swap(nRows, nCols);
        layout = (layout == RM) ? CM : RM; 
    }
        
    
    // Override operators
    float& operator()(uint32_t i, uint32_t j){
        if (i >= nRows || j >= nCols)
            throw std::out_of_range("Index out of range");
        return data[i * nCols + j];
    }

    const float& operator()(uint32_t i, uint32_t j) const{
        if (i >= nRows || j >= nCols)
            throw std::out_of_range("Index out of range");
        return data[i * nCols + j];
    }
    
    // GPUMatrix& operator=(const GPUMatrix& other){
    //     if (this != &other){
    //         nRows = other.nRows;
    //         nCols = other.nCols;
    //         layout = other.layout;
    //         cudaFree(data);
    //         cudaMalloc(&data, nRows * nCols * sizeof(float));
    //         cudaMemcpyAsync(data, other.data, nRows * nCols * sizeof(float), cudaMemcpyDeviceToDevice);
    //     }
    //     return *this;
    // }

};

    