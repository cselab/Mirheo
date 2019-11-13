#pragma once

#ifdef __NVCC__
#define __HD__ __host__ __device__
#define __D__  __device__
#else
#define __HD__
#define __D__
#endif
