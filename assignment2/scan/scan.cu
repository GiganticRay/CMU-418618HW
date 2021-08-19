#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"
#include "CudaError.cu"

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/* example for nextPow2
// 以指数的速率将低位覆盖以1
#include <iostream>
#include <bitset>
using namespace std;

int nextPow2(int n)
{
    n--;
    cout << "n--\t\t\t" << bitset<32>(n) << endl;
    n |= n >> 1;
    cout << "n >> 1 \t\t" << bitset<32>(n) << endl;
    n |= n >> 2;
    cout << "n >> 2 \t\t" << bitset<32>(n) << endl;
    n |= n >> 4;
    cout << "n >> 4 \t\t" << bitset<32>(n) << endl;
    n |= n >> 8;
    cout << "n >> 8 \t\t" << bitset<32>(n) << endl;
    n |= n >> 16;
    cout << "n >> 16 \t" << bitset<32>(n) << endl;
    n++;
    cout << "n++ \t\t" << bitset<32>(n) << endl;
    return n;
}

int main() {
    int n = 96;
    cout << "\t\t\t" << bitset<32>(n) << endl;
    cout << nextPow2(n) << endl;
}
*/


// implement parallel part of up_sweep phase
__global__ void kernel_scan_up_sweep(int* device_array, int length, int step){
    int index =     blockIdx.x*blockDim.x + threadIdx.x;
    int grid_size = gridDim.x*blockDim.x;
    for (int i = index+1; i*step-1 < length; i+=grid_size)
    {
        int log_sum = ceil(log2(i)) + ceil(log2(step));
        if(log_sum > 31){
            return;
        }
        device_array[i*step-1] += device_array[i*step - step/2 - 1]; 
    }
}

// implement parallel part of down_sweep phase
__global__ void kernel_scan_down_sweep(int* device_array, int length, int step){
    int index =     blockIdx.x*blockDim.x + threadIdx.x;
    int grid_size = gridDim.x*blockDim.x;
    for (int i = index+1; i*step-1 < length; i+=grid_size)
    {
        // as for i or step individually, it can't overflow. And of course the index of array can't overflow at all, so we could judge the log-sum of i and step
        // this process is so-called boundary-processing.
        int log_sum = ceil(log2(i)) + ceil(log2(step));
        if(log_sum > 31){
            return;
        }
        int tmp = device_array[i*step - step/2 - 1];
        device_array[i*step - step/2 - 1] = device_array[i*step-1];
        device_array[i*step-1] += tmp;
    }
    
}

// assign the value of element whose index is n to v
__global__ void kernel_assign(int *device_array, int n, int v){
    device_array[n] = v;
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
    length = nextPow2(length);
    cudaMemcpy(device_result, device_start, length*sizeof(int), cudaMemcpyDeviceToDevice);

    int grid_size   = 108;
    int block_size  = 1024;

    // 1. up_sweep parallel phase
    printf("Starting up-sweep phase: \n");
    for(int level=1; level<log2(length); level++){
        printf("up-sweep phase\tlevel=%d\n", level);
        int step = pow(2, level);

        kernel_scan_up_sweep<<<grid_size, block_size>>>(device_result, length, step);
        checkLastCuda();
    }

    // 2. set the last element of mid-result to 0.`
    // NOTICE!!!!!!: manipulate device memory in host code will invoke error:"Segmentation fault (core dumped)". Instead I invoke a single kernel function to modified the last element to 0.
    // device_start[length-1] = 0;
    kernel_assign<<<1, 1>>>(device_result, length-1, 0);

    // 3. down_sweep parallel phase
    printf("Starting down-sweep phase: \n");
    // ATTENTION: check out the process graph located in Figure 3. the level of down-sweep is 1 more than the up-sweep phase.
    printf("length: %d\n", length);
    for(int level=log2(length); level>=1; level--){
        printf("down-sweep phase\tlevel=%d\n", level);
        int step = pow(2, level);

        kernel_scan_down_sweep<<<grid_size, block_size>>>(device_result, length, step);
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

/* 
    function: set repeat logic step 1: parallel set token[i] = 1 if device_input[i+1] == device_input[i], otherwise token[i] = 0.
*/
__global__ void Kernel_Set_Token(int* device_input, int length, int* device_output){
    int step    = gridDim.x * blockDim.x;
    int threadIDX   = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = threadIDX; i < length-1; i+=step)
    {
        if(device_input[i] == device_input[i+1]){
            device_output[i] = 1;
        }else{
            device_output[i] = 0;
        }

        if(i == length-2){
            device_output[length-1] = 0;
        }
    }
}

/*
    function: set_repeat logic step 3: parallel set device_result[index[i-1 for index[i] != index[i-1]]] = i-1.
*/
__global__ void Kernel_Get_Repeat_From_Index(int * device_input, int length, int* device_output){
    int step    = gridDim.x * blockDim.x;
    int threadIDX   = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = threadIDX+1; i < length; i+=step)
    {
        if(device_input[i] != device_input[i-1]){
            device_output[device_input[i-1]] = i-1;
        }
    }
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */    

    /* logic
        1. parallel set token[i] = 1 if device_input[i+1] == device_input[i], otherwise token[i] = 0.
        2. exclusive prefix_add scan token to object_index.
        3. parallel set device_result[index[i-1 for index[i] != index[i-1]]] = i-1. 
       ex:
        input:  1 1 2 2 2 3 3 4 5 5
        token:  1 0 1 1 0 1 0 0 1 0
        index:  0 1 1 2 3 3 4 4 4 5
        result: result[1-1] = 1-1 = 0
                result[2-1] = 3-1 = 2
                result[3-1] = 4-1 = 3
                result[4-1] = 6-1 = 5
                result[5-1] = 9-1 = 8

    */
    int grid_size   = 108;
    int block_size  = 1024;
    int* device_token, * device_index;
    cudaMalloc((void **)&device_token, nextPow2(length) * sizeof(int));
    cudaMalloc((void **)&device_index, nextPow2(length) * sizeof(int));
    // step 1
    Kernel_Set_Token<<<grid_size, block_size>>>(device_input, length, device_token);
    // step 2
    cudaScan(device_token, device_token+length, device_index);
    // step 3
    Kernel_Get_Repeat_From_Index<<<grid_size, block_size>>>(device_index, length, device_output);
    int repeat_num = 0;
    cudaMemcpy(&repeat_num, device_index+length-1, sizeof(int), cudaMemcpyDeviceToHost);
    return repeat_num;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    printf("---------------------------------------------------------\n");
    printf("current active GPU\n");
    int curr_deviceId;
    cudaGetDevice(&curr_deviceId);
    cudaDeviceProp curr_props;
    cudaGetDeviceProperties(&curr_props, curr_deviceId);
    cudaGetDeviceProperties(&curr_props, curr_deviceId);
    printf("curr_Device name: %s\n", curr_props.name);
    printf("   SMs:        %d\n", curr_props.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
            static_cast<float>(curr_props.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", curr_props.major, curr_props.minor);


    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
