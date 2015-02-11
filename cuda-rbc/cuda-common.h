/*
 *  cuda-common.h
 *  ctc phenix
 *
 *  Created by Dmitry Alexeev on Nov 20, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <cstdio>
#include <vector_types.h>

//__device__ inline
//double __shfl_down(double var, unsigned int srcLane, int width=32)
//{
//	int2 a = *reinterpret_cast<int2*>(&var);
//	a.x = __shfl_down(a.x, srcLane, width);
//	a.y = __shfl_down(a.y, srcLane, width);
//	return *reinterpret_cast<double*>(&a);
//}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__inline__ __device__ float warpReduceSum(float val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val += __shfl_down(val, offset);
	}
	return val;
}

__inline__ __device__ float2 warpReduceSum(float2 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x += __shfl_down(val.x, offset);
		val.y += __shfl_down(val.y, offset);
	}
	return val;
}

__inline__ __device__ float3 warpReduceSum(float3 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x += __shfl_down(val.x, offset);
		val.y += __shfl_down(val.y, offset);
		val.z += __shfl_down(val.z, offset);
	}
	return val;
}


__inline__ __device__ float warpReduceMax(float val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val = max(__shfl_down(val, offset), val);
	}
	return val;
}

__inline__ __device__ float2 warpReduceMax(float2 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = max(__shfl_down(val.x, offset), val.x);
		val.y = max(__shfl_down(val.y, offset), val.y);
	}
	return val;
}

__inline__ __device__ float3 warpReduceMax(float3 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = max(__shfl_down(val.x, offset), val.x);
		val.y = max(__shfl_down(val.y, offset), val.y);
		val.z = max(__shfl_down(val.z, offset), val.z);
	}
	return val;
}


__inline__ __device__ float warpReduceMin(float val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val = min(__shfl_down(val, offset), val);
	}
	return val;
}

__inline__ __device__ float2 warpReduceMin(float2 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = min(__shfl_down(val.x, offset), val.x);
		val.y = min(__shfl_down(val.y, offset), val.y);
	}
	return val;
}

__inline__ __device__ float3 warpReduceMin(float3 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = min(__shfl_down(val.x, offset), val.x);
		val.y = min(__shfl_down(val.y, offset), val.y);
		val.z = min(__shfl_down(val.z, offset), val.z);
	}
	return val;
}
