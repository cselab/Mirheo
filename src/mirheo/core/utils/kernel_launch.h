#pragma once

#include <mirheo/core/logger.h>

namespace mirheo
{

/// \return \c true if the given 1D cuda block is non empty
inline bool isValid_nBlocks(int blocks)
{
    return blocks > 0;
}

/// \return \c true if the given 1D cuda block is non empty
inline bool isValid_nBlocks(dim3 blocks)
{
    return blocks.x > 0 && blocks.y > 0 && blocks.z > 0;
}

/** can be used inside MIRHEO_SAFE_KERNEL_LAUNCH,
    e.g. if the kernel has tmplate arguments; example:

    \code{.cpp}
    MIRHEO_SAFE_KERNEL_LAUNCH(
        kernelWithTemplates<T1 COMMA T2 COMMA T3>,
        blocks, threads, shmem, stream, args);
    \endcode
 */
#define COMMA ,


/** \brief Wrapper to launch device kernels

    All kernels in Mirheo must call this wrapper.
    This allows to automatically add a corresponding debug log entry.
    Furthermore, for high debug mode >= 9, the host will wait for the
    kernel to finish after each call.

    Empty kernels (empty grid) are skipped.
 */
#define MIRHEO_SAFE_KERNEL_LAUNCH(kernel, blocks, threads, shmem, stream, ...) \
    do {                                                                \
        if (isValid_nBlocks(blocks))                                    \
        {                                                               \
            debug4("Launching kernel "#kernel);                         \
            kernel <<< blocks, threads, shmem, stream >>> ( __VA_ARGS__ ); \
            if (logger.getDebugLvl() >= 9)                              \
            {                                                           \
                CUDA_Check( cudaStreamSynchronize(stream) );            \
                CUDA_Check( cudaPeekAtLastError() );                    \
            }                                                           \
        }                                                               \
        else                                                            \
        {                                                               \
            debug4("Kernel "#kernel" not launched, grid is empty");     \
        }                                                               \
    } while (0)

/// Macros shorthands.
#define SAFE_KERNEL_LAUNCH  MIRHEO_SAFE_KERNEL_LAUNCH

} // namespace mirheo
