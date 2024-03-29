#pragma once
#include <iostream>
#include "nccl.h"
#include "matmul.cuh"
#include "cuda_runtime.h"
#include <cub/cub.cuh>
#include <cublas_v2.h>

using namespace std;
#define str(s) to_string(s)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUBLASCHECK(cmd) do {                       \
  cublasStatus_t res = cmd;                         \
  if (res != CUBLAS_STATUS_SUCCESS) {               \
    printf("Failed, CUBLAS error %s:%d '%s'\n",     \
        __FILE__,__LINE__,cublasGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

///////////////////// Softmax //////////////////////
struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

/**
 * @brief Softmax kernel with online calculation of normalization term.
 * See https://arxiv.org/pdf/1805.02867.pdf
*/
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
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

}


////////////////////////////// Matrix //////////////////////////////
enum class MatrixLayout {
	RowMajor = 0,
	ColumnMajor = 1,
};
static constexpr MatrixLayout RM = MatrixLayout::RowMajor;
static constexpr MatrixLayout CM = MatrixLayout::ColumnMajor;

template <typename T>
void device_copy(T* src, T* dst, uint32_t n, cudaStream_t stream=nullptr){
    // There seems to be no speed difference between Memcpy and MemcpyPeer...some say the only diff is we can only use Memcpy when UVA is on so that runtime can figure out the devices
    // (https://stackoverflow.com/questions/22694518/what-is-the-difference-between-cudamemcpy-and-cudamemcpypeer-for-p2p-copy) 
    // Otherwise we have to specify them using MemcpyPeer
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDefault, stream));
}


class GPUMatrix{
    private:
        int device = 0;
        
    public:
        uint32_t nRows;
        uint32_t nCols;
        float *data;
        MatrixLayout layout;

        //Default constructor
        GPUMatrix() : nRows(0), nCols(0), data(nullptr), layout(RM) {}

        // Constructor with device array data
        // Set device to the target device you want the data to be on 
        GPUMatrix(uint32_t nRows, uint32_t nCols, float* new_data, MatrixLayout layout=RM, cudaStream_t stream=nullptr, int device=-1)
            : nRows(nRows), nCols(nCols), layout(layout), device(device) {

                int context;
                CHECK_CUDA_ERROR(cudaGetDevice(&context));

                if (this->device == -1) // No device specified
                    CHECK_CUDA_ERROR(cudaGetDevice(&this->device));
                    
                // Switch to target device
                if (this->device != context)
                    CHECK_CUDA_ERROR(cudaSetDevice(this->device));

                // Check if data is a device array
                cudaPointerAttributes attr;
                cudaPointerGetAttributes(&attr, new_data);
                
                // Case 1: data is a host array
                if (attr.type == cudaMemoryTypeHost || attr.type == cudaMemoryTypeUnregistered){ // Unregistered is normal new-allocated array
                    CHECK_CUDA_ERROR(cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream));
                    // Debug here
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(data, new_data, nRows * nCols * sizeof(float), cudaMemcpyDefault, stream));
                }
                // case 2: data is on a different device
                else if (attr.device != device){
                    CHECK_CUDA_ERROR(cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream));
                    device_copy(new_data, data, nRows * nCols, stream);
                }
                // Case 3: data is on the same device
                else{
                    data = new_data;
                }

                // Restore device context
                CHECK_CUDA_ERROR(cudaSetDevice(context));
            }
        
        // Constructor without data
        GPUMatrix(uint32_t nRows, uint32_t nCols, MatrixLayout layout=RM, cudaStream_t stream=nullptr, int device=-1)
            : nRows(nRows), nCols(nCols), layout(layout), device(device) {
                CHECK_CUDA_ERROR(cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream));
                // Set to current device if not specified
                if (this->device == -1)
                    CHECK_CUDA_ERROR(cudaGetDevice(&this->device));
            }

        ~GPUMatrix(){
            if (this->has_data())
                CHECK_CUDA_ERROR(cudaFree(data));
            data = nullptr;
        }   
        
        // Transpose the matrix in-place
        void T(cublasHandle_t handle){
            if (!this->has_data())
                throw std::invalid_argument("Matrix has no underlying data");

            // Check if data and context are on the same device
            int context;
            CHECK_CUDA_ERROR(cudaGetDevice(&context));
            if (context != device)
                CHECK_CUDA_ERROR(cudaSetDevice(device));

            transpose(data, data, nRows, nCols, handle); // Transpose 
            // Reverse layout
            std::swap(nRows, nCols);
            layout = (layout == RM) ? CM : RM; 

            // Restore device context
            if (context != device)
                CHECK_CUDA_ERROR(cudaSetDevice(context));
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
                int context;
                CHECK_CUDA_ERROR(cudaGetDevice(&context));
                CHECK_CUDA_ERROR(cudaSetDevice(new_device));

                // Copy to new device
                float *new_data;
                CHECK_CUDA_ERROR(cudaMallocAsync(&new_data, nRows * nCols * sizeof(float), stream));
                device_copy(data, new_data, nRows * nCols, stream);
                CHECK_CUDA_ERROR(cudaFreeAsync(data, stream));
                data = new_data;

                // Restore device context
                CHECK_CUDA_ERROR(cudaSetDevice(context));

            }
        }
        
        // Allocate memory of the specified size
        void init_data(uint32_t nRows = 0, uint32_t nCols = 0, MatrixLayout layout=RM, cudaStream_t stream=nullptr){
            if (this->has_data())
                printf("Skipping data init as it's already there\n");
            if (nRows == 0 || nCols == 0){
                nRows = this->nRows;
                nCols = this->nCols;
            }
            this->layout = layout;
            
            // Save current device context
            int context = 0;
            CHECK_CUDA_ERROR(cudaGetDevice(&context));
            CHECK_CUDA_ERROR(cudaSetDevice(this->device));

            CHECK_CUDA_ERROR(cudaMallocAsync(&data, nRows * nCols * sizeof(float), stream)); // Allocate memory
            CHECK_CUDA_ERROR(cudaSetDevice(context)); // Restore context

        }


        /////////////////////// Override operators ///////////////////////
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
        
        // Copy constructor
        GPUMatrix(const GPUMatrix& other)
            : nRows(other.nRows), nCols(other.nCols), layout(other.layout), device(other.device) {
            // Allocate new memory on the correct device
            int context = 0;
            CHECK_CUDA_ERROR(cudaGetDevice(&context));
            CHECK_CUDA_ERROR(cudaSetDevice(other.device));
            CHECK_CUDA_ERROR(cudaMallocAsync(&data, other.nRows * other.nCols * sizeof(float), nullptr));
            CHECK_CUDA_ERROR(cudaSetDevice(context)); // Restore context
            // Copy data
            device_copy(other.data, data, other.nRows * other.nCols, nullptr);
        }

        // Copy operator
        GPUMatrix& operator=(const GPUMatrix& other){

            if (this != &other){
                // No data or shape mismatch
                if (data != nullptr && (nRows != other.nRows || nCols != other.nCols || device != other.device)) {
                    // Deallocate current data if it exists
                    cudaFreeAsync(data, nullptr);
                    data = nullptr;
                }
                
                if (data == nullptr) {
                    int context = 0;
                    CHECK_CUDA_ERROR(cudaGetDevice(&context));
                    CHECK_CUDA_ERROR(cudaSetDevice(other.device));
                    // Allocate new memory on the correct device
                    cudaMallocAsync(&data, other.nRows * other.nCols * sizeof(float), nullptr);
                    device = other.device;
                    
                    CHECK_CUDA_ERROR(cudaSetDevice(context)); // Restore context
                }

                device_copy(other.data, data, other.nRows * other.nCols, nullptr);
                // else if (data != other.data){
                //     // Data exists and not the same

                //     // You should make sure sizes match before assigning
                //     if (nRows != other.nRows || nCols != other.nCols){
                //         string message = "Can't assign when matrix dimensions don't match: (" + str(nRows) + ", " + str(nCols)
                //             + ") vs (" + str(other.nRows) + ", " + str(other.nCols) + ")";
                //         throw std::invalid_argument(message);
                //     }
                    
                //     // If on different devices,  copy new data to current device
                //     if (device != other.device){
                //         device_copy(other.data, data, other.nRows * other.nCols, nullptr);
                //     }                   
                // }else{
                //     // Same data, do nothing
                //     return *this;
                // }

                // Assign other attributes
                nRows = other.nRows;
                nCols = other.nCols;
                layout = other.layout;
                device = other.device;
            }
            
            return *this;
        }


        // Move constructor
        GPUMatrix(GPUMatrix&& other) noexcept
            : data(other.data), nRows(other.nRows), nCols(other.nCols), layout(other.layout), device(other.device) {
            // Transfer ownership and invalidate the source
            other.data = nullptr;
            other.nRows = 0;
            other.nCols = 0;
            other.device = -1; // Assuming -1 is an invalid device ID
        }

        // Move assignment operator
        GPUMatrix& operator=(GPUMatrix&& other) noexcept {
            if (this != &other) {
                // Free existing resource
                // Assuming you have a mechanism to safely free GPU memory
                cudaFreeAsync(data, nullptr);

                // Transfer ownership
                data = other.data;
                nRows = other.nRows;
                nCols = other.nCols;
                layout = other.layout;
                device = other.device;

                // Invalidate the source
                other.data = nullptr;
                other.nRows = 0;
                other.nCols = 0;
                other.device = -1; // Assuming -1 is an invalid device ID
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

////////////////////////////// MLP //////////////////////////////

class MLP{
public:
    // model config
    uint32_t num_layers;
    uint32_t num_devices;
    bool tp_enabled = false;

    // Matrix data
    MatrixDims* full_layer_dims; // Full dims without TP splitting
    MatrixDims* full_activation_dims;
    vector<vector <GPUMatrix> > dev_weights;
    vector<vector <GPUMatrix> > dev_activations;
    // NCCL & cublas
    ncclComm_t* nccl_comm;
    cudaStream_t* nccl_streams;
    std::map <int, cublasHandle_t> device_handles;

    //////////////////////// Constructors ////////////////////////
    /**
     * @brief Default constructor
    */
    MLP() : num_layers(0), full_layer_dims(nullptr), num_devices(0) {}
    
    /**
     * @brief No weight constructor
    */
    MLP(uint32_t num_layers, MatrixDims* full_layer_dims, uint32_t num_devices)
       : num_layers(num_layers), full_layer_dims(full_layer_dims), num_devices(num_devices)
        {
            dev_weights.resize(1); 
            dev_weights[0].reserve(num_layers); // Pre-allocate but skip constructor
            uint32_t last_nCols = full_layer_dims[0].nCols;

            // Allocate mem for the whole model on device 0
            for (uint32_t i = 0; i < num_layers; i++){
                uint32_t nRows = full_layer_dims[i].nRows;
                uint32_t nCols = full_layer_dims[i].nCols;

                // layer dim check
                if (nRows != last_nCols)
                    throw invalid_argument(string(__FILE__) + ":" + str(__LINE__) + ": 'Invalid weight dimensions: nCols (" + str(last_nCols) +
                        ") from last layer does not match nRows (" + str(nRows) + ") of layer' " + str(i));

                last_nCols = nCols;
                dev_weights[0].emplace_back(nRows, nCols, RM);
            }
        }

    /**
    * @brief Construct a new MLP object with weights
    */
    MLP(uint32_t num_layers, MatrixDims* full_layer_dims, uint32_t num_devices, float** mat_weights)
        : num_layers(num_layers), full_layer_dims(full_layer_dims), num_devices(num_devices)
        {
            dev_weights.resize(1); 
            dev_weights[0].reserve(num_layers); // Pre-allocate 

            uint32_t last_nCols = full_layer_dims[0].nRows; 
            for (uint32_t i = 0; i < num_layers; i++){
                uint32_t nRows = full_layer_dims[i].nRows;
                uint32_t nCols = full_layer_dims[i].nCols;

                // layer dim check
                if (nRows != last_nCols)
                    throw invalid_argument(string(__FILE__) + ":" + str(__LINE__) + ": Invalid weight dimensions: nCols (" + str(last_nCols) +
                        ") from last layer does not match nRows (" + str(nRows) + ") of layer " + str(i));
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
        // destroy handles
        for (auto& [device, handle] : device_handles){
            cublasDestroy(handle);
        }

    }

    // Single GPU forward pass
    void forward(GPUMatrix& input, const GPUMatrix& output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        
        // Use default device
        dev_activations.resize(1); 
        uint32_t dev = 0; 
        CHECK_CUDA_ERROR(cudaSetDevice(dev));

        // Allocate activation buffers
        make_activation_buffers(batch_size, 1);

        uint32_t in_dim = input.nCols;
        uint32_t out_dim = dev_weights[0][0].nCols;
        
        matmul_rect_relu(input.data, dev_weights[dev][0].data, dev_activations[dev][0].data,
                batch_size, in_dim, out_dim, stream);

        uint32_t i;
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

    // Tensor parallelism forward. Must call enable_tp() first.
    void forward_tp(const GPUMatrix& input,  GPUMatrix& output, cudaStream_t stream=nullptr, const uint32_t batch_size=32){
        // check that num layers = 3
        if (num_layers != 3){
            throw invalid_argument(string(__FILE__) + ":" + str(__LINE__) + ": Tensor parallelism is only supported for 3-layer MLPs\n");
        }
        if (num_devices < 2){
            throw invalid_argument(string(__FILE__) + ":" + str(__LINE__) + ": Tensor parallelism requires at least 2 devices\n");
        }

        // Set up buffers and NCCL for tensor parallelism
        if (tp_enabled == false)
            enable_tp(batch_size, stream);
        
        //First layer
        uint32_t in_dim = input.nCols;
        uint32_t out_dim = dev_weights[1][0].nCols; // scattered dim on non-default devices
        for (uint32_t dev = 0; dev < num_devices; dev++){
            CHECK_CUDA_ERROR(cudaSetDevice(dev));
            matmul_rect_relu(input.data, dev_weights[dev][0].data, input.data, 
                batch_size, in_dim, out_dim, stream);
        }

        //Second layer
        in_dim = out_dim;
        out_dim = dev_weights[1][1].nCols;
        for (uint32_t dev = 0; dev < num_devices; dev++){
            CHECK_CUDA_ERROR(cudaSetDevice(dev));
            matmul_rect_relu(dev_activations[dev][0].data, dev_weights[dev][1].data, dev_activations[dev][1].data,
                batch_size, in_dim, out_dim, stream);
        }
        reduce_activations(1);

        
        // Transpose the relu->reduced input , split it column-wise to cut the computations by num_devices
        // Here we can split the input and then reduce because the kernel doesn't apply a non-linear activation
        in_dim = out_dim;
        out_dim = dev_weights[1][2].nCols;
        this->transpose(dev_activations[0][1]); // -> (out_dim, batch_size)
        for (uint32_t dev = 1; dev < num_devices; dev++ ){
            uint32_t nRows = dev_activations[dev][1].nRows;
            uint32_t nCols = dev_activations[dev][1].nCols;
            uint32_t offset = dev * nCols * nRows;
            nRows += (dev == num_devices - 1) ? dev_activations[0][1].nRows % num_devices : 0; // Handle non-divisible case
            
            dev_activations[dev][1] = GPUMatrix(nRows, nCols, dev_activations[0][1].data + offset, CM, stream, dev); // Async copy (nCols, nRows) from (out_dim, batch_size)
            this->transpose(dev_activations[dev][1]); // -> (nRows, nCols)
        }
        
        // Third layer
        for (uint32_t dev = 0; dev < num_devices; dev++){
            CHECK_CUDA_ERROR(cudaSetDevice(dev));
            matmul(dev_activations[dev][1].data, dev_weights[dev][2].data, dev_activations[dev][2].data,
                batch_size, in_dim, out_dim, static_cast<cudaStream_t>(stream));
        }
        // Reduce and softmax
        reduce_activations(2);
        call_softmax(dev_activations[0][2].data, output.data, output.nCols, batch_size, static_cast<cudaStream_t>(stream));

        cudaSetDevice(0); // Switch back to default device
        output = dev_activations[0][2];

    }

    /**
     * @brief Get the activation dims based on weight dims and alloc buffers
    */
    void make_activation_buffers(uint32_t batch_size, uint32_t num_devices){
        full_activation_dims = new MatrixDims[num_layers];
        
        // Compute full dims
        for (uint32_t i = 0; i < num_layers; i++){
            full_activation_dims[i].nRows = batch_size;
            full_activation_dims[i].nCols = full_layer_dims[i].nCols;
        }
        // Allocate buffers
        int context;
        CHECK_CUDA_ERROR(cudaGetDevice(&context));

        dev_activations.resize(num_devices); 
        for (uint32_t dev = 0; dev < num_devices; dev++){
            CHECK_CUDA_ERROR(cudaSetDevice(dev));

            dev_activations[dev].reserve(num_layers); 
            for (uint32_t i = 0; i < num_layers; i++){
                
                // Reuses the pre-computed dims of scattered device weights
                uint32_t nRows = batch_size;
                uint32_t nCols = dev_weights[dev][i].nCols;
                // Allocate buffers
                dev_activations[dev].emplace_back(nRows, nCols, RM, nullptr, dev);
            }
        }
       
        // Restore device context
        CHECK_CUDA_ERROR(cudaSetDevice(context));
    }
    

    /**
     * @brief Allocate and scatter weight chunks from device 0 to each device 
    */
    void scatter_to_devices(vector<GPUMatrix>& src_data, vector<vector<GPUMatrix>>& device_data, MatrixDims* dims, cudaStream_t stream = nullptr) {
        // Pre-allocate memory for device_data vectors
        device_data.resize(num_devices); 

        // Save device context
        int context;
        CHECK_CUDA_ERROR(cudaGetDevice(&context));

        // Allocate buffers on each device
        for (uint32_t dev = 0; dev < num_devices; dev++) {
            device_data[dev].reserve(num_layers);

            for (uint32_t j = 0; j < num_layers; j++) {
                CHECK_CUDA_ERROR(cudaSetDevice(dev));
            
                // Allocate splitted buffers on device but reserve space for the full data on the default device 0 
                // Compute and update in-place buffer dimensions
                uint32_t nRows = dims[j].nRows;
                uint32_t nCols = dims[j].nCols;
                // Scatter dims to devices
                if (dev != 0){
                    // Handle non-divisible case: last device gets the remainder
                    nRows = nRows / num_devices + (dev == num_devices - 1) ? nRows % num_devices : 0; 
                    // nCols = nCols / num_devices;
                }
                    
                // scatter source data to each device
                if (src_data[j].has_data()) {
                    uint32_t offset = dev * nCols;
                    // Debug here !!
                    device_data[dev].emplace_back(nRows, nCols, src_data[j].data + offset, src_data[j].layout, stream, dev);
                    
                    if (device_data[dev][j].layout == CM)
                        this->transpose(device_data[dev][j]); // Transpose back the scattered weight to row-major 

                // No data, just allocate
                } else {
                    device_data[dev].emplace_back(nRows, nCols, RM, stream, dev);
                }
            }
        }
        // Restore device context
        CHECK_CUDA_ERROR(cudaSetDevice(context));
    }

    /** 
    * @brief Enabling tensor parallelism
    * Recipe for tensor parallel: 
    * Split 1st weight matrix column-wise, 2nd row-wise and 3rd layer and its input row-wise
    * Our way of transpose->scatter is the same as Megatron-LM: see https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/core/tensor_parallel/utils.py#L11
    **/
    void enable_tp(int batch_size, cudaStream_t stream=nullptr){
        tp_enabled = true;
        int context;
        CHECK_CUDA_ERROR(cudaGetDevice(&context));
        
        // Transpose in-place
        this->transpose(dev_weights[0][0]);
        
        // vector<GPUMatrix> dev_weights_copy = dev_weights[0]; // Create a copy of dev_weights[0]
        // scatter_to_devices(dev_weights_copy, dev_weights, full_layer_dims, stream); // Scatter weight chunks to each device
        scatter_to_devices(dev_weights[0], dev_weights, full_layer_dims, stream); // Scatter weight chunks to each device
        make_activation_buffers(batch_size, this->num_devices); // Allocate activation buffers on each device

        //Init NCCL context & group & streams
        int device_ids[num_devices];
        for (uint32_t i = 0; i < num_devices; i++){
            device_ids[i] = i;
            CHECK_CUDA_ERROR(cudaSetDevice(i));
            CHECK_CUDA_ERROR(cudaStreamCreate(&nccl_streams[i]));
        }
        NCCLCHECK(ncclCommInitAll(nccl_comm, num_devices, device_ids));
        
        CHECK_CUDA_ERROR(cudaSetDevice(context)); // Restore context
    }


    // Gather the weights but might not be neccesary 
    // Memcpy from other devices
    void disable_tp(){
        tp_enabled = false;

        this->transpose(dev_weights[0][0]); // Transpose back
        // Delete weights on non-default devices
        for (uint32_t dev = 1; dev < num_devices; dev++){
                dev_weights[dev].clear();
                dev_activations[dev].clear();
        }
        for (uint32_t i = 0; i < num_devices; i++){
            CHECK_CUDA_ERROR(cudaStreamDestroy(nccl_streams[i]));
        }
        // Clear Cublas handles
        for (auto& [device, handle] : device_handles){
            cublasDestroy(handle);
        }
    }


    // Reduce activations from all devices to device 0
    void reduce_activations(int layer, bool all_reduce=false){
        // Save device context
        int context;
        CHECK_CUDA_ERROR(cudaGetDevice(&context));

        NCCLCHECK(ncclGroupStart());
        // Malloc receive buffer on default device
        // All-reduce (see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
        int recv_rank = 0;
        for (uint32_t dev = 1; dev < num_devices; ++dev)
        {
            const void* send_buff = (const void*)dev_activations[0][layer].data;
            void* recv_buff = (void*)dev_activations[dev][layer].data;
            uint32_t size = dev_activations[dev][layer].nRows * dev_activations[dev][layer].nCols;
            
            if (all_reduce)
                NCCLCHECK(ncclAllReduce(send_buff, recv_buff, size, ncclFloat, ncclSum, nccl_comm[dev], nccl_streams[dev]));
            else
                NCCLCHECK(ncclReduce(send_buff, recv_buff, size, ncclFloat, ncclSum, recv_rank, nccl_comm[dev], nccl_streams[dev]));
        }

        // Sync and wait for reduction results
        for (uint32_t dev = 0; dev < num_devices; ++dev)
        {
            CHECK_CUDA_ERROR(cudaSetDevice(dev));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(nccl_streams[dev]));
        }
        NCCLCHECK(ncclGroupEnd());

        // Restore device context
        CHECK_CUDA_ERROR(cudaSetDevice(context));
    }

    void transpose(GPUMatrix& matrix){
        // Check if handle exists
        if (device_handles.find(matrix.get_device()) == device_handles.end()){
            cublasCreate(&device_handles[matrix.get_device()]);
        }
        matrix.T(device_handles[matrix.get_device()]);   
    }

};