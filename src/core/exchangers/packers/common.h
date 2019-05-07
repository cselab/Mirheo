#pragma once

#include "interface.h"

#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>


namespace CommonPackerKernels
{

template <typename T>
__global__ static void updateOffsets(int n, const int *sizes, size_t *offsetsBytes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;
    
    size_t sz = Packer::getPackedSize<T>(sizes[i]);
    offsetsBytes[i] += sz;
}

template <typename T>
__global__ static void updateOffsetsObjects(int n, int objSize, const int *sizes, size_t *offsetsBytes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;
    
    size_t sz = Packer::getPackedSize<T>(sizes[i] * objSize);
    offsetsBytes[i] += sz;
}

} // namespace CommonPackerKernels


template <typename T>
static void updateOffsets(int n, const int *sizes, size_t *offsetsBytes, cudaStream_t stream)
{
    constexpr int nthreads = 32;

    SAFE_KERNEL_LAUNCH(
        CommonPackerKernels::updateOffsets<T>,
        getNblocks(n, nthreads), nthreads, 0, stream,
        n, sizes, offsetsBytes);
}

template <typename T>
static void updateOffsetsObjects(int n, int objSize, const int *sizes, size_t *offsetsBytes, cudaStream_t stream)
{
    constexpr int nthreads = 32;

    SAFE_KERNEL_LAUNCH(
        CommonPackerKernels::updateOffsetsObjects<T>,
        getNblocks(n, nthreads), nthreads, 0, stream,
        n, objSize, sizes, offsetsBytes);
}
