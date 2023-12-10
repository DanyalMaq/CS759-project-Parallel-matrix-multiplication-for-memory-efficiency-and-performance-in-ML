#pragma once
#include <iostream>
using namespace std;

///////////////////// Activations //////////////////////
template <typename T>
__host__ __device__ T relu(T val) {
	return (T)max((float)val, 0.0f);
}


// TODO: upgrade to online softmax
template <uint32_t N>
__host__ __device__ inline float softmax(const float vals[N], uint32_t idx) {
    
	float total = 0;

	for (uint32_t i = 0; i < N; ++i) {
		total += expf(vals[i]);
	}

	return expf(vals[idx]) / total;
}

///////////////////// Matrix //////////////////////

enum class MatrixLayout {
	RowMajor = 0,
	ColumnMajor = 1,
};
static constexpr MatrixLayout RM = MatrixLayout::RowMajor;
static constexpr MatrixLayout CM = MatrixLayout::ColumnMajor;

template <typename T>
void device_copy(T* src, T* dst, uint32_t n, int src_device, int dst_device, cudaStream_t stream=nullptr){
    int canAccessPeer;
    cudaDeviceCanAccessPeer(&canAccessPeer, src_device, dst_device);
    // There seems to be no speed difference...some posts say the only diff is we can use Memcpy only when UVA is enabled so that runtime can figure out the devices
    // (https://stackoverflow.com/questions/22694518/what-is-the-difference-between-cudamemcpy-and-cudamemcpypeer-for-p2p-copy) 
    // Otherwise we have to specify them using MemcpyPeer
    if (canAccessPeer)
        cudaMemcpyPeerAsync(dst, dst_device, src, src_device, n * sizeof(T), stream);
    else
        cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}


class GPUMatrix{
    public:
        uint32_t nRows;
        uint32_t nCols;
        float *data;
        int device = 0;
        MatrixLayout layout;
    // Default constructor
    GPUMatrix() : nRows(0), nCols(0), data(nullptr), layout(RM) {}

    // Constructor with device array data
    GPUMatrix(uint32_t nRows, uint32_t nCols, float* new_data, MatrixLayout layout=RM, cudaStream_t stream=nullptr, int device=-1)
        : nRows(nRows), nCols(nCols), layout(layout), device(device) {
            // Set to the current device if not specified
            if (device == -1)
                cudaGetDevice(&device);

            // Check if data is a device array
            cudaPointerAttributes attr;
            cudaPointerGetAttributes(&attr, new_data);
            
            // Case 1: data is a host array
            if (attr.type == cudaMemoryTypeHost){
                cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream);
                cudaMemcpyAsync(new_data, data, nRows * nCols * sizeof(float), cudaMemcpyHostToDevice, stream);
            }
            // case 2: data is on a different device
            else if (attr.device != device){
                cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream);
                device_copy(new_data, data, nRows * nCols, attr.device, device, stream);
            }
            // Case 3: data is on the same device
            else{
                data = new_data;
            }
        }
    
    // Constructor without data
    GPUMatrix(uint32_t nRows, uint32_t nCols, MatrixLayout layout=RM, cudaStream_t stream=nullptr, int device=-1)
        : nRows(nRows), nCols(nCols), layout(layout), device(device) {
            cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream);
            // Set to current context's device if not specified
            if (device == -1)
                cudaGetDevice(&device);
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
        
    bool has_data() const{
        return data != nullptr;
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
                    device_copy(other.data, data, other.nRows * other.nCols, other.device, device, nullptr);
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
    LayerDim** device_layer_dims;
    GPUMatrix* weights;
    GPUMatrix** device_weights;
    GPUMatrix** device_middle_buffers;
    
    // Allocate (and possibly initialize) weight chunks on each device
    void alloc_device_weights(){
        device_weights = new GPUMatrix*[num_devices];

        for (int i = 1; i < num_devices; i++){
            device_weights[i - 1] = new GPUMatrix[num_layers];
            for (int j = 0; j < num_layers; j++){
            cudaSetDevice(i);

            uint32_t nRows = layer_dims[j].nRows;
            uint32_t nCols = layer_dims[j].nCols;
            uint32_t nRows_per_device = i != num_devices - 1 ? nRows / num_devices : nRows / num_devices + nRows % num_devices; // Handle non-divisible case
            uint32_t nCols_per_device = nCols / num_devices;
            
            if (weights != nullptr and weights[j].has_data())
                // Perform p2p copy on the chunk 
                device_weights[i - 1][j](nRows_per_device, nCols_per_device, weights[j].data, RM, nullptr, i);

            device_weights[i - 1][j](nRows_per_device, nCols_per_device);
            }
        }
    }


    // Default constructor
    MLP() : num_layers(0), layer_dims(nullptr), num_devices(0), in_dim(0), weights(nullptr) {}

    
    // No weight constructor
    MLP(uint32_t num_layers, LayerDim* layer_dims, uint32_t num_devices, uint32_t in_dim = 784)
       : num_layers(num_layers), layer_dims(layer_dims), num_devices(num_devices), in_dim(in_dim)
        {
            weights = new GPUMatrix[num_layers];
            uint32_t last_nCols = in_dim;

            // Allocate mem for the whole model on device 0
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

            alloc_device_weights();

        }

    // Constructor with weights
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
    
    // Destructor
    ~MLP(){
        delete[] weights;
    }

    // Enabling tensor parallelism
    void enable_tp(){
        tp_enabled = true;

        // Split 1st weight matrix row-wise, 2nd column-wise and 3rd layer row-wise
        weights[1].T(); // Change the layout and transpose

        // Then alloc device arrays on each device and do p2p copy1
        device_weights = new GPUMatrix*[num_devices];

        for (int i = 0; i < num_layers; i++){
            for (int j = 1; j < num_devices; j++){
                cudaSetDevice(j);
                uint32_t nRows = layer_dims[i].nRows;
                uint32_t nCols = layer_dims[i].nCols;
                // handle non-divisible case
                uint32_t nRows_per_device = j < num_devices - 1 ? nRows / num_devices : nRows / num_devices + nRows % num_devices;
                uint32_t nCols_per_device = nCols / num_devices;
                device_weights[j] = new GPUMatrix(nRows_per_device, nCols_per_device, RM);

            }
        }
        // TODO: Init NCCL context & group
    }

    // Gather the weights but might not be neccesary 
    // Memcpy from other devices
    // void disable_tp(){}

    // void alloc_buffers()

    // Single GPU forward pass
    void forward(GPUMatrix& input, const GPUMatrix& output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        
        // Check if buffer exists & batch size matches
        // TODO: Allocate intermediate buffers for each layer based on batch size
        device_middle_buffers = new GPUMatrix*[1];
        cudaSetDevice(0);
        uint32_t nRows;
        uint32_t nCols;
        for (int i = 0; i < num_layers; i++){
            nRows = layer_dims[i].nRows;
            nCols = layer_dims[i].nCols;
            device_middle_buffers[i] = new GPUMatrix(nRows, nCols, RM);
        }

        for (int i = 0; i < num_layers; i++){
            matmul(weights[0].data, input.data, output->data, weights[i].nRows, weights[i].nCols, input->data, stream);
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
        // TODO: Alloc immediate buffers for each layer on each device based on batch size

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