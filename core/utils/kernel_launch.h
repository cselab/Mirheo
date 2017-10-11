#pragma once

#if 1
#include <core/logger.h>
#else

#define debug4(...)

struct Logger
{
	int getDebugLvl() { return 10 };
} logger;

#endif


template<class T>
static inline bool isValid_nBlocks(T blocks);

template <> bool isValid_nBlocks<int>(int blocks)
{
	return blocks > 0;
}

template <> bool isValid_nBlocks<dim3>(dim3 blocks)
{
	return blocks.x > 0 && blocks.y > 0 && blocks.z > 0;
}


#define  SAFE_KERNEL_LAUNCH(kernel, blocks, threads, shmem, stream, ...)                \
do {                                                                                    \
	if (isValid_nBlocks(blocks))                                                        \
	{                                                                                   \
		debug4("Launching kernel "#kernel);                                             \
		kernel <<< blocks, threads, shmem, stream >>> (__VA_ARGS__);                    \
		if (logger.getDebugLvl() >= 9)                                                  \
		{                                                                               \
			cudaStreamSynchronize(stream);                                              \
			CUDA_Check( cudaPeekAtLastError() );                                        \
		}                                                                               \
	}                                                                                   \
	else                                                                                \
	{                                                                                   \
		debug4("Kernel "#kernel" not launched, grid is empty");                         \
	}                                                                                   \
} while (0)
