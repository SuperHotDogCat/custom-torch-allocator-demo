#include <cuda_runtime_api.h>
#include <stdlib.h>

extern "C" {

void *custom_malloc(ssize_t size, int device, cudaStream_t stream) {
    void *ptr;
    #ifdef MANAGED
    cudaMallocManaged(&ptr, size);
    #elif CPU
    cudaMallocHost(&ptr, size);
    #else
    cudaMallocAsync(&ptr, size, stream);
    #endif
    return ptr;
}

void custom_free(void *ptr, ssize_t size, int device, cudaStream_t stream) {
    #ifdef MANAGED
    cudaFree(ptr);
    #elif CPU
    cudaFreeHost(ptr);
    #else
    cudaFreeAsync(ptr, stream);
    #endif
}

}