// Author: Nic Olsen, Jason Zhou

# pragma once
#include <iostream>
using namespace std;
const unsigned int TILE_WIDTH = 32; // Tile size of shared memory

__host__ void matmul(float *A, float *B, float *C,
                    int nRowsA, int nColsA, int nColsB, 
                    cudaEvent_t start = nullptr, cudaEvent_t stop = nullptr, cudaStream_t stream = nullptr);
__host__ void matmul(float* A, float* B, float* C,
                    int nRowsA, int nColsA, int nColsB, cudaStream_t stream = nullptr);
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
        int device = 0;
        MatrixLayout layout;
    // Default constructor
    GPUMatrix() : nRows(0), nCols(0), data(nullptr), layout(RM) {}

    // Constructor with data
    GPUMatrix(uint32_t nRows, uint32_t nCols, float* data, MatrixLayout layout=RM)
        : nRows(nRows), nCols(nCols), data(data), layout(layout) {}
    
    // Constructor without data
    GPUMatrix(uint32_t nRows, uint32_t nCols, MatrixLayout layout=RM, cudaStream_t stream=nullptr)
        : nRows(nRows), nCols(nCols), layout(layout) {
            cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream);
            // get device
            cudaPointerAttributes attr;
            cudaPointerGetAttributes(&attr, data);
            device = attr.device;
        }
        

    ~GPUMatrix(){
        cudaFree(data);
        data = nullptr;
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
    

    GPUMatrix& operator=(const GPUMatrix& other){
        // Handle cuda array data

        if (this != &other){
            // No data, just assign
            if (data == nullptr){
                data = other.data;
            } else if (data != other.data){
                // Data exists and not the same
                
                // Free old data
                cudaFreeAsync(data, nullptr);

                // Handle different sizes
                if (nRows != other.nRows || nCols != other.nCols){
                    printf("Warning: assigning to a matrix with different dimensions\n");
                }
                
                // Handle different devices
                if (device != other.device){
                    // copy new data to old device
                    cudaMallocAsync(&data, other.nRows * other.nCols * sizeof(float), nullptr);
                    // check if p2p is on
                    int canAccessPeer;
                    cudaDeviceCanAccessPeer(&canAccessPeer, device, other.device);
                    if (canAccessPeer)
                        cudaMemcpyPeerAsync(data, device, other.data, other.device, other.nRows * other.nCols * sizeof(float), nullptr);
                    else
                        cudaMemcpyAsync(data, other.data, other.nRows * other.nCols * sizeof(float), cudaMemcpyDeviceToDevice, nullptr);
                }                   
            }else{
                // Same data, do nothing
                return *this;
            }

            // Assign other attributes
            nRows = other.nRows;
            nCols = other.nCols;
            layout = other.layout;
            device = other.device;
            layout = other.layout;
        }
        return *this;
    }
};


struct LayerDim {
    uint32_t nRows;
    uint32_t nCols;
};


class MLP{
public:
    uint32_t num_layers;
    uint32_t num_devices;
    uint32_t in_dim;
    bool tp_enabled = false;

    LayerDim* layer_dims;
    GPUMatrix* weights;
    GPUMatrix* device_weights;
    GPUMatrix* device_middle_buffers;
    
    // Default constructor
    MLP() : num_layers(0), layer_dims(nullptr), num_devices(0), in_dim(0), weights(nullptr) {}

    // No weight constructor
    MLP(uint32_t num_layers, LayerDim* layer_dims, uint32_t num_devices, uint32_t in_dim = 784)
       : num_layers(num_layers), layer_dims(layer_dims), num_devices(num_devices), in_dim(in_dim)
        {
            weights = new GPUMatrix[num_layers];
            uint32_t last_nCols = in_dim;
            for (int i = 0; i < num_layers; i++){
                uint32_t nRows = layer_dims[i].nRows;
                uint32_t nCols = layer_dims[i].nCols;

                // layer dim check
                if (nRows != last_nCols)
                    throw std::invalid_argument("Invalid weight dimensions: nCols (" + to_string(last_nCols) +
                        ") from last layer does not match nRows (" + to_string(nRows) + ") of layer" + to_string(i));
                last_nCols = nCols;

                weights[i] = *(new GPUMatrix(nRows, nCols, RM));
            }
        }

    // Weight constructor
    MLP(uint32_t num_layers, LayerDim* layer_dims, uint32_t num_devices, float** mat_weights, uint32_t in_dim = 784)
        : num_layers(num_layers), layer_dims(layer_dims), num_devices(num_devices), in_dim(in_dim)
        {
            weights = new GPUMatrix[num_layers];
            uint32_t last_nCols = in_dim;
            for (int i = 0; i < num_layers; i++){
                uint32_t nRows = layer_dims[i].nRows;
                uint32_t nCols = layer_dims[i].nCols;

                // layer dim check
                if (nRows != last_nCols)
                    throw std::invalid_argument("Invalid weight dimensions: nCols (" + to_string(last_nCols) +
                        ") from last layer does not match nRows (" + to_string(nRows) + ") of layer" + to_string(i));
                last_nCols = nCols;

                weights[i] = *(new GPUMatrix(nRows, nCols, mat_weights[i], RM));
            }
        }
    

    ~MLP(){
        delete[] weights;
    }

    // Enabling tensor parallelism
    void enable_tp(){
        tp_enabled = true;
        weights[0].T(); // the first matrix needs to be split row-wise

        // Split 1st weight matrix row-wise, 2nd column-wise and 3rd layer row-wise
        // Then alloc device arrays on each device and do p2p copy1
        device_weights = new GPUMatrix[num_devices];
        for (int i = 0; i < num_devices; i++){
            // TODO: handle non-divisible case
            device_weights[i] = *(new GPUMatrix(weights[0]->nRows / num_devices, weights[0]->nCols, RM));
            device_weights[i] = *(new GPUMatrix(weights[1]->nRows, weights[1]->nCols / num_devices, CM));
            device_weights[i] = *(new GPUMatrix(weights[2]->nRows / num_devices, weights[2]->nCols, RM));
        }
        
        // TODO: Init NCCL context & group
    }

    // Gather the weights but might not be neccesary 
    // Memcpy from other devices
    // void disable_tp(){}

    // void alloc_buffers()

    // Single GPU forward pass
    void forward(float* input, const float* output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        
        // Check if buffer exists & batch size matches
        // Alloc intermediate buffers for each layer based on batch size

        for (int i = 0; i < num_layers; i++){
            matmul(weights[i]->data, input, output, weights[i]->nRows, weights[i]->nCols, input->nCols, stream);
            // input = output;
        }
    }

    
    void forward_tp(float* input, const float* output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        // check that num layers = 3
        if (num_layers != 3){
            printf("Tensor parallelism is only supported for 3-layer MLPs\n");
            return;
        }
        // Check if buffer exists & batch size matches
        // Alloc immediate buffers for each layer on each device based on batch size

        // For the first two layers, just copy the full input to each device
        // For layer 3, transpose and split the input column-wise, then copy to each device
        
        // for each layer
            // for each device
                // Set device 
                // Transpose input?
                // memcpyp2pAsync
                // matmul
                // ReLU

        // NCCL all-reduce and softmax
        
    }
        
};