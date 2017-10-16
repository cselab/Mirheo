#pragma once

struct DomainInfo
{
	float3 globalSize, globalStart, localSize;

	__forceinline__ __host__ __device__ float3 local2global(float3 x) const
	{
		return x + globalStart + 0.5f * localSize;
	}
	__forceinline__ __host__ __device__ float3 global2local(float3 x) const
	{
		return x - globalStart - 0.5f * localSize;
	}
};
