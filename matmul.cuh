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
        if (this != &other){
            nRows = other.nRows;
            nCols = other.nCols;
            layout = other.layout;
            
            if (data != nullptr && data != other.data){
                // check if they are on the same device
                cudaPointerAttributes attr1, attr2;
                cudaPointerGetAttributes(&attr1, data);
                cudaPointerGetAttributes(&attr2, other.data);
                if (attr1.device != attr2.device){
                    cudaFree(data);
                    cudaMallocAsync(&data, nRows * nCols * sizeof(float), nullptr);
                    // check if p2p is on
                    int canAccessPeer;
                    cudaDeviceCanAccessPeer(&canAccessPeer, attr1.device, attr2.device);
                    if (canAccessPeer)
                        cudaMemcpyPeerAsync(data, attr1.device, other.data, attr2.device, nRows * nCols * sizeof(float), nullptr);
                    else
                        cudaMemcpyAsync(data, other.data, nRows * nCols * sizeof(float), cudaMemcpyDeviceToDevice, nullptr);

                }else{
                    data = other.data;
                }

            }
        }
        return *this;
    }

};


class MLP{
public:
    uint32_t num_layers;
    GPUMatrix* weights;
    uint32_t num_devices;
    GPUMatrix* device_weights;
    bool tp_enabled = false;
    GPUMatrix* device_middle_buffers;
    
    // No weight constructor
    MLP(uint32_t num_layers, std::tuple<uint32_t, uint32_t>* layer_sizes, uint32_t num_devices){
        this->num_layers = num_layers;
        this->layer_sizes = layer_sizes;
        this->num_devices = num_devices;
        weights = new GPUMatrix*[num_layers];

        uint32_t last_nCols;
        for (int i = 0; i < num_layers; i++){
            uint32_t nRows = std::get<0>(layer_sizes[i]);
            uint32_t nCols = std::get<1>(layer_sizes[i]);

            // layer dim check
            if (i !=0 && nRows != last_nCols)
                throw std::invalid_argument("Invalid weight dimensions: nCols (" + to_string(last_nCols) 
                    ") from last layer does not match nRows (" + + to_string(nRows) + ") of layer" + to_string(i));
            else
                last_nCols = nCols;

            weights[i] = new GPUMatrix(nRows, nCols, RM);
        }
    }

    // Weight constructor
    MLP(uint32_t num_layers, std::tuple<uint32_t, uint32_t>* layer_sizes, uint32_t num_devices, float** weights){
        this->num_layers = num_layers;
        this->layer_sizes = layer_sizes;
        this->num_devices = num_devices;
        this->weights = new GPUMatrix*[num_layers];
        
        uint32_t last_nCols; 
        for (int i = 0; i < num_layers; i++){
            uint32_t nRows = std::get<0>(layer_sizes[i]);
            uint32_t nCols = std::get<1>(layer_sizes[i]);

            // layer dim check
            if (i !=0 && nRows != last_nCols)
                throw std::invalid_argument("Invalid weight dimensions: nCols (" + to_string(last_nCols) 
                    ") from last layer does not match nRows (" + + to_string(nRows) + ") of layer" + to_string(i));
            else
                last_nCols = nCols;

            this->weights[i] = new GPUMatrix(nRows, nCols, weights[i], RM);
        }
    }

    ~MLP(){
        for (int i = 0; i < num_layers; i++){
            delete weights[i];
        }
        delete[] weights;
    }

    // Enabling tensor parallelism
    void enable_tp(){
        tp_enabled = true;
        weights[0]->T(); // the first matrix needs to be split row-wise

        // Split 1st weight matrix row-wise, 2nd column-wise and 3rd layer row-wise
        // Then alloc device arrays on each device and do p2p copy1

        for (int i = 0; i < num_devices; i++){
            device_weights[i] = new GPUMatrix*[num_layers];
            // TODO: handle non-divisible case
            device_weights[i][0] = new GPUMatrix(weights[0]->nRows / num_devices, weights[0]->nCols, RM);
            device_weights[i][1] = new GPUMatrix(weights[1]->nRows, weights[1]->nCols / num_devices, CM);
            device_weights[i][2] = new GPUMatrix(weights[2]->nRows / num_devices, weights[2]->nCols, RM);
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