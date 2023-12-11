#pragma once
#include <iostream>
#include "nccl.h"

using namespace std;

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)



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
    // There seems to be no speed difference...some say the only diff is we can only use Memcpy when UVA is on so that runtime can figure out the devices
    // (https://stackoverflow.com/questions/22694518/what-is-the-difference-between-cudamemcpy-and-cudamemcpypeer-for-p2p-copy) 
    // Otherwise we have to specify them using MemcpyPeer
    if (canAccessPeer)
        cudaMemcpyPeerAsync(dst, dst_device, src, src_device, n * sizeof(T), stream);
    else
        cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}


class GPUMatrix{
    private:
        int device = 0;
        
    public:
        uint32_t nRows;
        uint32_t nCols;
        float *data;
        MatrixLayout layout;

        // Default constructor
        GPUMatrix() : nRows(0), nCols(0), data(nullptr), layout(RM) {}

        // Constructor with device array data
        // Set device to the target device you want the data to be on 
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
        
        // Transpose the matrix in-place
        void T(){
            transpose(data, data, nRows, nCols);
            // swap nRows and nCols and reverse layout
            std::swap(nRows, nCols);
            layout = (layout == RM) ? CM : RM; 
        }
        
        ///////////////// Setters and getters /////////////////
        bool has_data() const{
            return data != nullptr;
        }
        
        int get_device() const{
            return device;
        }

        // Change device and transfer data
        void set_device(int new_device, cudaStream_t stream=nullptr){
            if (new_device != device){
                // Save device context like stack
                int current_context;
                cudaGetDevice(&current_context);
                cudaSetDevice(new_device);

                // Malloc on new device and copy data
                float *new_data;
                cudaMallocAsync(&new_data, nRows * nCols * sizeof(float), stream);
                device_copy(data, new_data, nRows * nCols, device, new_device, stream);
                cudaFreeAsync(data, stream);
                data = new_data;

                // Restore device context
                cudaSetDevice(current_context);

            }
        }

        ///////////////// Override operators //////////////////
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
        

        // Assign (data transfer) operator
        GPUMatrix& operator=(const GPUMatrix& other){

            if (this != &other){
                // No data, just assign
                if (data == nullptr){
                    data = other.data;
                } else if (data != other.data){
                    // Data exists and not the same

                    // You should make sure sizes match before assigning
                    if (nRows != other.nRows || nCols != other.nCols){
                        string message = "Can't assign when matrix dimensions don't match: (" + to_string(nRows) + ", " + to_string(nCols)
                            + ") vs (" + to_string(other.nRows) + ", " + to_string(other.nCols) + ")";
                        throw std::invalid_argument(message);
                    }
                    
                    // If on different devices,  copy new data to current device
                    if (device != other.device){
                        device_copy(other.data, data, other.nRows * other.nCols, other.device, device, nullptr);
                    }                   
                }else{
                    // Same data, do nothing
                    return *this;
                }

                // Assign other attributes
                // nRows = other.nRows;
                // nCols = other.nCols;
                layout = other.layout;
                device = other.device;
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
    // model config
    uint32_t num_layers;
    uint32_t num_devices;
    uint32_t in_dim;
    bool tp_enabled = false;

    // Matrix data
    LayerDim* layer_dims;
    LayerDim** device_layer_dims;
    vector<GPUMatrix> weights;
    vector<vector <GPUMatrix> > device_weights;
    vector<vector <GPUMatrix> > device_middle_buffers;
    
    // NCCL
    ncclComm_t* nccl_comm;

    // Allocate on and scatter chunks from device 0 to each device 
    void scatter_to_device_chunks(vector<vector<GPUMatrix>>& device_data, LayerDim* layer_dims, vector<GPUMatrix>* src_data = nullptr, cudaStream_t stream = nullptr) {
        // Pre-allocate memory for device_data vectors
        device_data.resize(num_devices - 1); 

        // Save device context
        int current_context;
        cudaGetDevice(&current_context);

        // Allocate buffers on each device
        for (int dev = 1; dev < num_devices; dev++) {
            device_data[dev - 1].reserve(num_layers);

            for (int j = 0; j < num_layers; j++) {
                cudaSetDevice(dev);
                // Compute chunk dimensions
                uint32_t nRows = layer_dims[j].nRows;
                uint32_t nCols = layer_dims[j].nCols;
                // Handle non-divisible case
                uint32_t nRows_per_device = dev != num_devices - 1 ? nRows / num_devices : nRows / num_devices + nRows % num_devices;
                uint32_t nCols_per_device = nCols / num_devices;

                // If source data is provided, scatter it to each device
                if (src_data != nullptr && (*src_data)[j].has_data()) {
                    device_data[dev - 1].emplace_back(nRows_per_device, nCols_per_device, (*src_data)[j].data, (*src_data)[j].layout, stream, dev);
                // No data provided, just allocate
                } else {
                    device_data[dev - 1].emplace_back(nRows_per_device, nCols_per_device, RM, stream, dev);
                }
            }
        }
    // Restore device context
    cudaSetDevice(current_context);
}

    // Default constructor
    MLP() : num_layers(0), layer_dims(nullptr), num_devices(0), in_dim(0) {}

    
    // No weight constructor
    MLP(uint32_t num_layers, LayerDim* layer_dims, uint32_t num_devices, uint32_t in_dim = 784)
       : num_layers(num_layers), layer_dims(layer_dims), num_devices(num_devices), in_dim(in_dim)
        {
            weights.reserve(num_layers); // Pre-allocate but avoid default-construction
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
                weights.emplace_back(nRows, nCols, RM);
            }

        }

    // Constructor with weights
    MLP(uint32_t num_layers, LayerDim* layer_dims, uint32_t num_devices, float** mat_weights, uint32_t in_dim = 784)
        : num_layers(num_layers), layer_dims(layer_dims), num_devices(num_devices), in_dim(in_dim)
        {
            weights.reserve(num_layers); // Pre-allocate but avoid default-construction

            uint32_t last_nCols = in_dim;
            for (int i = 0; i < num_layers; i++){
                uint32_t nRows = layer_dims[i].nRows;
                uint32_t nCols = layer_dims[i].nCols;

                // layer dim check
                if (nRows != last_nCols)
                    throw std::invalid_argument("Invalid weight dimensions: nCols (" + to_string(last_nCols) +
                        ") from last layer does not match nRows (" + to_string(nRows) + ") of layer" + to_string(i));
                last_nCols = nCols;
                weights.emplace_back(nRows, nCols, mat_weights[i], RM);
            }
        }
    
    // Destructor
    ~MLP(){
        delete[] layer_dims;
        delete[] device_layer_dims;
        delete[] nccl_comm;
        device_weights.clear();
        device_middle_buffers.clear();

    }

    // Enabling tensor parallelism
    // Recipe for tensor parallel: 
    // Split 1st weight matrix row-wise, 2nd column-wise and 3rd layer row-wise
    void enable_tp(cudaStream_t stream=nullptr){
        tp_enabled = true;

        weights[1].T(); // Transpose in-place
        scatter_to_device_chunks(device_weights, layer_dims, &weights, stream);

        // TODO: Init NCCL context & group
        int device_ids[num_devices];
        for (int i = 0; i < num_devices; i++)
            device_ids[i] = i;
        NCCLCHECK(ncclCommInitAll(nccl_comm, num_devices, device_ids));
    }

    // Gather the weights but might not be neccesary 
    // Memcpy from other devices
    // void disable_tp(){}

    // void alloc_buffers()

    // Single GPU forward pass
    void forward(GPUMatrix& input, const GPUMatrix& output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        
        // Check if buffer exists & batch size matches
        // TODO: Allocate intermediate buffers for each layer based on batch size
        device_middle_buffers.resize(1); // Single device

        cudaSetDevice(0);
        uint32_t nRows;
        uint32_t nCols;
        for (int i = 0; i < num_layers; i++){
            nRows = layer_dims[i].nRows;
            nCols = layer_dims[i].nCols;
            device_middle_buffers[0].emplace_back(nRows, nCols, RM, stream, 0);
        }

        for (int i = 0; i < num_layers - 1; i++){
            matmul(input.data, weights[i].data, device_middle_buffers[0][i].data,
                input.nRows, input.nCols, weights[i].nRows, weights[i].nCols, stream);

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
                // Memcpyp2p Async
                // matmul
                // ReLU

        // NCCL all-reduce and softmax
        
    }
        
};