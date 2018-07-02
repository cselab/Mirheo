#pragma once

#include <core/utils/cuda_common.h>

struct DomainInfo
{
	float3 globalSize, globalStart, localSize;

	inline __host__ __device__ float3 local2global(float3 x) const
	{
		return x + globalStart + 0.5f * localSize;
	}
	inline __host__ __device__ float3 global2local(float3 x) const
	{
		return x - globalStart - 0.5f * localSize;
	}
};
