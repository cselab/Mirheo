#pragma once

#ifdef __NVCC__
#define __HD__ __host__ __device__
#define __D__  __device__
#else

#include <cuda_runtime.h>
#include <vector_types.h>

#define __HD__
#define __D__
#endif

