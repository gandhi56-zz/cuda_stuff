#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "VecAdd.h"

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i=0; i<N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("cpu %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
            break;
        }   
    }
    if (match)
        printf("Arrays match.\n\n");
}

void initializeData(float* ip, int n){
    time_t t;
    srand((unsigned)time(&t));
    for (int i =0 ; i < n; ++i){
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

void vectorAddOnHost(float* a, float* b, float* c, const int n){
    for (int i = 0; i < n; ++i){
        c[i] = a[i] + b[i];
    }
}

__global__
void vectorAddOnDevice(float* a, float* b, float* c, const int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int run(void){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // set up data size for vectors
    const int numElements = 32000;
    printf("Vector size = %d\n", numElements);

    // malloc host memory
    size_t numBytes = numElements * sizeof(float);

    float *h_a, *h_b, *hostRef, *gpuRef;
    h_a = (float*)malloc(numBytes);
    h_b = (float*)malloc(numBytes);
    hostRef = (float*)malloc(numBytes);
    gpuRef = (float*)malloc(numBytes);

    // initialize data on host
    initializeData(h_a, numElements);
    initializeData(h_b, numElements);

    memset(hostRef, 0, numBytes);
    memset(gpuRef, 0, numBytes);

    // malloc device global memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, numBytes);
    cudaMalloc((float**)&d_b, numBytes);
    cudaMalloc((float**)&d_c, numBytes);

    // transfer data from host to device
    cudaMemcpy(d_a, h_a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, numBytes, cudaMemcpyHostToDevice);

    // invoke kernel on device
    dim3 block(min(numElements, 1024)); // assuming 1024 is max number of threads per block
    dim3 grid(ceil((float)numElements / block.x));
    vectorAddOnDevice<<<grid, block>>>(d_a, d_b, d_c, numElements);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    // copy result back to host
    cudaMemcpy(gpuRef, d_c, numBytes, cudaMemcpyDeviceToHost);
    
    // add vectors on host
    vectorAddOnHost(h_a, h_b, hostRef, numElements);

    // check results
    checkResult(hostRef, gpuRef, numElements);

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free host memory
    free(h_a);
    free(h_b);
    free(hostRef);
    free(gpuRef);

    return 0;
}









