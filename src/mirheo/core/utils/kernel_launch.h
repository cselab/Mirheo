#pragma once

#include <mirheo/core/logger.h>

inline bool isValid_nBlocks(int blocks)
{
    return blocks > 0;
}

inline bool isValid_nBlocks(dim3 blocks)
{
    return blocks.x > 0 && blocks.y > 0 && blocks.z > 0;
}

#define COMMA ,

#define  SAFE_KERNEL_LAUNCH(kernel, blocks, threads, shmem, stream, ...)      \
do {                                                                          \
    if (isValid_nBlocks(blocks))                                              \
    {                                                                         \
        debug4("Launching kernel "#kernel);                                   \
        kernel <<< blocks, threads, shmem, stream >>> ( __VA_ARGS__ );        \
        if (logger.getDebugLvl() >= 9)                                        \
        {                                                                     \
            CUDA_Check( cudaStreamSynchronize(stream) );                      \
            CUDA_Check( cudaPeekAtLastError() );                              \
        }                                                                     \
    }                                                                         \
    else                                                                      \
    {                                                                         \
        debug4("Kernel "#kernel" not launched, grid is empty");               \
    }                                                                         \
} while (0)
