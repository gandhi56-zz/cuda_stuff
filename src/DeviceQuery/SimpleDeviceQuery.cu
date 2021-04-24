#include <cuda_runtime.h>
#include "DeviceQuery.h"
#include <stdio.h>

int SimpleDeviceQuery::run(){
    int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, iDev);

    printf("Device %d: %s\n", iDev, iProp.name);
    printf("Number of multiprocessors: %d\n", iProp.multiProcessorCount);
    printf("Total amount of Constant Memory: %4.2f KB\n", iProp.totalConstMem / 1024.0);
    printf("Total amount of Shared memory per block: %4.2f KB\n", iProp.sharedMemPerBlock/1024.0);
    printf("Total number of registers per block: %d\n", iProp.regsPerBlock);
    printf("Warp size: %d\n", iProp.warpSize);
    printf("Maximum number of threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor / 32);
    return 0;
}