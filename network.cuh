#pragma once
#include <iostream>
#include "nccl.h"
#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using namespace std;

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

///////////////////// Softmax //////////////////////
struct __align__(8) MD
{
    float m;
    float d;
};

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void online_softmax(
    const float * __restrict x,
    float * __restrict y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x and y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        MD new_elem;
        new_elem.m = x[elem_id];
        new_elem.d = 1.0F;
        md_partial = reduce_md_op(md_partial, new_elem);
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (thread_id == 0)
        md_total = md;
    __syncthreads();

    float d_total_inverse = __fdividef(1.0F, md_total.d);
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id] - md_total.m) * d_total_inverse;
}

void call_softmax(const float* x, float* y, uint32_t nCols, uint32_t batch_size, cudaStream_t stream=nullptr)
{
    uint32_t THREADBLOCK_SIZE = nCols / 8; // Heuristic to have at least 8 iterations of the loop
    if (THREADBLOCK_SIZE >= 256)
        online_softmax<256><<<batch_size , 256, 0, stream>>>(x, y, nCols);
    else if (THREADBLOCK_SIZE >= 128)
        online_softmax<128><<<batch_size, 128, 0, stream>>>(x, y, nCols);
    else if (THREADBLOCK_SIZE >= 64)
        online_softmax<64><<<batch_size, 64, 0, stream>>>(x, y, nCols);
    else if (THREADBLOCK_SIZE >= 32)
        online_softmax<32><<<batch_size, 32, 0, stream>>>(x, y, nCols);

}


//////////////////////// Matrix //////////////////////////
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
                int context;
                cudaGetDevice(&context);

                if (device == -1)
                    cudaGetDevice(&device);
                // Switch to target device to allocate memory
                if (device != context)
                    cudaSetDevice(device);

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

                // Restore device context
                cudaSetDevice(context);
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
            // Check if data and context are on the same device
            int context;
            cudaGetDevice(&context);
            if (context != device)
                cudaSetDevice(device);

            transpose(data, data, nRows, nCols); // Transpose 
            // Reverse layout
            std::swap(nRows, nCols);
            layout = (layout == RM) ? CM : RM; 

            // Restore device context
            if (context != device)
                cudaSetDevice(context);
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

struct MatrixDims {
    uint32_t nRows;
    uint32_t nCols;

    // Default constructor
    MatrixDims(uint32_t rows, uint32_t cols) : nRows(rows), nCols(cols) {};
    MatrixDims() : nRows(0), nCols(0) {};
};


class MLP{
public:
    // model config
    uint32_t num_layers;
    uint32_t num_devices;
    uint32_t in_dim;
    bool tp_enabled = false;

    // Matrix data
    MatrixDims* full_layer_dims; // Full dims without TP splitting
    MatrixDims* full_activation_dims;
    vector<vector <GPUMatrix> > dev_weights;
    vector<vector <GPUMatrix> > dev_activations;
    // NCCL
    ncclComm_t* nccl_comm;
    cudaStream_t* nccl_streams;

    // Get the activation dims based on the weight dims
    void make_activation_buffers(uint32_t batch_size, uint32_t num_devices){
        full_activation_dims = new MatrixDims[num_layers];
        
        // Compute full dims
        for (int i = 0; i < num_layers; i++){
            full_activation_dims[i].nRows = batch_size;
            full_activation_dims[i].nCols = full_layer_dims[i].nCols;
        }
        // Allocate buffers
        int context;
        cudaGetDevice(&context);

        dev_activations.resize(num_devices); 
        for (int dev = 0; dev < num_devices; dev++){
            cudaSetDevice(dev);

            dev_activations[dev].reserve(num_layers); 
            for (int i = 0; i < num_layers; i++){
                
                // Reuses the pre-computed dims of scattered device weights
                uint32_t nRows = batch_size;
                uint32_t nCols = dev_weights[dev][i].nCols;
                // Allocate buffers
                dev_activations[dev].emplace_back(nRows, nCols, RM, nullptr, dev);
            }
        }
       
        // Restore device context
        cudaSetDevice(context);
    }
    

    // Allocate on and scatter chunks from device 0 to each device 
    void scatter_to_devices(vector<vector<GPUMatrix>>& device_data, MatrixDims* data_dims, vector<GPUMatrix>* src_data = nullptr, cudaStream_t stream = nullptr) {
        // Pre-allocate memory for device_data vectors
        device_data.resize(num_devices); 

        // Save device context
        int current_context;
        cudaGetDevice(&current_context);

        // Allocate buffers on each device
        for (int dev = 0; dev < num_devices; dev++) {
            device_data[dev].reserve(num_layers);

            for (int j = 0; j < num_layers; j++) {
                cudaSetDevice(dev);
            
                // Allocate splitted buffers on device but reserve space for the full data on the default device 0 
                // Compute and update in-place buffer dimensions
                uint32_t nRows = data_dims[j].nRows;
                uint32_t nCols = data_dims[j].nCols;
                // Scatter dims to devices
                if (dev != 0){
                    // Handle non-divisible case: last device gets the remainder
                    nRows = nRows / num_devices + (dev == num_devices - 1) ? nRows % num_devices : 0; 
                    nCols = nCols / num_devices;
                }
                    
                // If source data is provided, scatter it to each device
                if (src_data != nullptr && (*src_data)[j].has_data()) {
                    uint32_t offset = dev * nCols;
                    device_data[dev].emplace_back(nRows, nCols, (*src_data)[j].data + offset, (*src_data)[j].layout, stream, dev);
                    
                    if (device_data[dev][j].layout == CM)
                        device_data[dev][j].T(); // Transpose back the scattered weight to row-major 

                // No data, just allocate
                } else {
                    device_data[dev - 1].emplace_back(nRows, nCols, RM, stream, dev);
                }
            }
        }
        // Restore device context
        cudaSetDevice(current_context);
    }

    // Default constructor
    MLP() : num_layers(0), full_layer_dims(nullptr), num_devices(0), in_dim(0) {}
    
    // No weight constructor
    MLP(uint32_t num_layers, MatrixDims* full_layer_dims, uint32_t num_devices)
       : num_layers(num_layers), full_layer_dims(full_layer_dims), num_devices(num_devices)
        {
            dev_weights.resize(1); 
            dev_weights[0].reserve(num_layers); // Pre-allocate but avoid default-construction
            uint32_t last_nCols = in_dim;

            // Allocate mem for the whole model on device 0
            for (int i = 0; i < num_layers; i++){
                uint32_t nRows = full_layer_dims[i].nRows;
                uint32_t nCols = full_layer_dims[i].nCols;

                // layer dim check
                if (nRows != last_nCols)
                    throw std::invalid_argument("Invalid weight dimensions: nCols (" + to_string(last_nCols) +
                        ") from last layer does not match nRows (" + to_string(nRows) + ") of layer" + to_string(i));

                last_nCols = nCols;
                dev_weights[0].emplace_back(nRows, nCols, RM);
            }

        }

    // Constructor with weights
    MLP(uint32_t num_layers, MatrixDims* full_layer_dims, uint32_t num_devices, float** mat_weights)
        : num_layers(num_layers), full_layer_dims(full_layer_dims), num_devices(num_devices), in_dim(in_dim)
        {
            dev_weights.resize(1); 
            dev_weights[0].reserve(num_layers); // Pre-allocate but avoid default-construction

            uint32_t last_nCols = in_dim;
            for (int i = 0; i < num_layers; i++){
                uint32_t nRows = full_layer_dims[i].nRows;
                uint32_t nCols = full_layer_dims[i].nCols;

                // layer dim check
                if (nRows != last_nCols)
                    throw std::invalid_argument("Invalid weight dimensions: nCols (" + to_string(last_nCols) +
                        ") from last layer does not match nRows (" + to_string(nRows) + ") of layer" + to_string(i));
                last_nCols = nCols;
                dev_weights[0].emplace_back(nRows, nCols, mat_weights[i], RM);
            }
        }
    
    // Destructor
    ~MLP(){
        delete[] full_layer_dims;
        delete[] nccl_comm;
        dev_weights.clear();
        dev_activations.clear();

    }

    // Enabling tensor parallelism
    // Recipe for tensor parallel: 
    // Split 1st weight matrix column-wise, 2nd row-wise and 3rd layer and input x row-wise
    void enable_tp(int batch_size, cudaStream_t stream=nullptr){
        tp_enabled = true;
        int context;
        cudaGetDevice(&context);

        dev_weights[0][0].T(); // Transpose in-place
        scatter_to_devices(dev_weights, full_layer_dims, &dev_weights[0], stream); // Scatter weight chunks to each device
        make_activation_buffers(batch_size, this->num_devices); // Allocate activation buffers on each device

        //Init NCCL context & group & streams
        int device_ids[num_devices];
        for (int i = 0; i < num_devices; i++){
            device_ids[i] = i;
            cudaSetDevice(i);
            cudaStreamCreate(&nccl_streams[i]);
        }
        NCCLCHECK(ncclCommInitAll(nccl_comm, num_devices, device_ids));
        
        cudaSetDevice(context); // Restore context
    }

    // Gather the weights but might not be neccesary 
    // Memcpy from other devices
    void disable_tp(){
        tp_enabled = false;

        dev_weights[0][0].T(); // Transpose back
        for (int dev = 1; dev < num_devices; dev++){
                dev_weights[dev].clear();
                dev_activations[dev].clear();
        }
        
        for (int i = 0; i < num_devices; i++){
            cudaStreamDestroy(nccl_streams[i]);
        }
    }


    void reduce_activations(int layer, bool all_reduce=false){
        // Save device context
        int context;
        cudaGetDevice(&context);

        NCCLCHECK(ncclGroupStart());
        // Malloc receive buffer on default device
        // All-reduce (see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
        for (int dev = 1; dev < num_devices; ++dev)
        {
            const void* send_buff = (const void*)dev_activations[0][layer].data;
            void* recv_buff = (void*)dev_activations[dev][layer].data;
            uint32_t size = dev_activations[dev][layer].nRows * dev_activations[dev][layer].nCols;
            if (all_reduce)
                NCCLCHECK(ncclAllReduce(send_buff, recv_buff, size, ncclFloat, ncclSum, nccl_comm[dev], nccl_streams[dev]));
            else
                NCCLCHECK(ncclReduce(send_buff, recv_buff, size, ncclFloat, ncclSum, nccl_comm[dev], nccl_streams[dev]));
        }

        // Sync and wait for reduction results
        for (int dev = 0; dev < num_devices; ++dev)
        {
            cudaSetDevice(dev);
            cudaStreamSynchronize(nccl_streams[dev]);
        }
        NCCLCHECK(ncclGroupEnd());

        // Restore device context
        cudaSetDevice(context);
    }


    // Single GPU forward pass
    void forward(GPUMatrix& input, const GPUMatrix& output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        
        // Use default device
        dev_activations.resize(1); 
        uint32_t dev = 0; 
        cudaSetDevice(dev);

        // Allocate activation buffers
        make_activation_buffers(batch_size, 1);

        uint32_t in_dim = input.nCols;
        uint32_t out_dim = dev_weights[0][0].nCols;
        
        matmul_rect_relu(input.data, dev_weights[dev][0].data, dev_activations[dev][0].data,
                batch_size, in_dim, out_dim, stream);

        int i;
        for (i = 1; i < num_layers - 1; i++){
            // TODO: must use kernel call wrapper for this 
            in_dim = out_dim;
            out_dim = dev_weights[dev][i].nCols;
            matmul_rect_relu(dev_activations[dev][i - 1].data, dev_weights[dev][i].data, dev_activations[dev][i].data,
                batch_size, in_dim, out_dim, stream);
        }

        // Final layer: linear + softmax
        in_dim = out_dim;
        out_dim = dev_weights[dev][i].nCols;
        matmul(dev_activations[dev][i].data, dev_weights[dev][i].data, output.data,
                batch_size, in_dim, out_dim, stream);
        call_softmax(dev_activations[dev][i].data, output.data, output.nCols, batch_size, stream);
        
    }

    
    void forward_tp(GPUMatrix& input, const GPUMatrix& output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        // check that num layers = 3
        if (num_layers != 3){
            throw std::invalid_argument("Tensor parallelism is only supported for 3-layer MLPs\n");
        }
        if (num_devices < 2){
            throw std::invalid_argument("Tensor parallelism requires at least 2 devices\n");
        }

        // Save device context
        int context; 
        cudaGetDevice(&context);

        // Set up buffers and NCCL for tensor parallelism
        if (tp_enabled == false)
            enable_tp(stream);
        
        //First layer
        uint32_t in_dim = input.nCols;
        uint32_t out_dim = dev_weights[1][0].nCols; // scattered dim on non-default devices
        for (int dev = 0; dev < num_devices; dev++){
            cudaSetDevice(dev);
            matmul_rect_relu(input.data, dev_weights[dev][0], input.data, 
                batch_size, in_dim, out_dim, stream);
        }

        //Second layer
        in_dim = out_dim;
        out_dim = dev_weights[1][1].nCols;
        for (int dev = 0; dev < num_devices; dev++){
            cudaSetDevice(dev);
            matmul_rect_relu(dev_activations[dev][0].data, dev_weights[dev][1], dev_activations[dev][1].data 
                batch_size, in_dim, out_dim, stream);
        }
        reduce_activations(1);

        
        // Transpose the relu->reduced input , split it column-wise and scatter to cut the computations by num_devices
        // Here we can split the input and then reduce because the kernel doesn't apply a non-linear activation
        in_dim = out_dim;
        out_dim = dev_weights[1][2].nCols;
        dev_activations[0][1].T(); // -> (out_dim, batch_size)
        for (int dev = 1; dev < num_devices; dev++ ){
            uint32_t nRows = dev_activations[dev][1].nRows;
            uint32_t nCols = dev_activations[dev][1].nCols;
            uint32_t offset = dev * nCols * nRows;
            nRows += (dev == num_devices - 1) ? dev_activations[0][1].nRows % num_devices : 0; // Handle non-divisible case
            uint32_t size = nCols * nRows;
            
            dev_activations[dev][1] = GPUMatrix(nRows, nCols, dev_activations[0][1].data + offset, CM, stream, dev); // Async copy (nCols, nRows) from (out_dim, batch_size)
            dev_activations[dev][1].T(); // -> (nRows, nCols)
        }
        
        // Third layer
        for (int dev = 0; dev < num_devices; dev++){
            cudaSetDevice(dev);
            matmul(dev_activations[dev][1].data, dev_weights[dev][2], dev_activations[dev][2].data,
                batch_size, in_dim, out_dim, stream);
        }
        // Reduce and softmax
        reduce_activations(2);
        call_softmax(dev_activations[0][2].data, output.data, output.nCols, batch_size, stream);

        cudaSetDevice(0);
        output = dev_activations[0][2];

        // Restore device context
        cudaSetDevice(context);

    }
        
};