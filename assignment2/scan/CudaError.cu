#include <stdio.h>
#include <assert.h>
#include <cuda.h>

// Notice: this function could only be used to detect Cuda function whose return value type is not void. 
// we could use cudaGetLastError() to detect such cuda function (ex, self_define kernel function.)
cudaError_t checkCuda(cudaError_t result){
    if(result != cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;

}

cudaError_t checkLastCuda(){
    cudaError_t result = cudaGetLastError();    // return the error from above
    if(result != cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }

    return result;
}