#pragma once
#include <stdio.h>

#define CHECK(call)                                                                     \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess)                                                           \
    {                                                                                   \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                                   \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));             \
        exit(1);                                                                        \
    }                                                                                   \
}

namespace VecAdd{
    int run(void);
};  // end namespace VecAdd