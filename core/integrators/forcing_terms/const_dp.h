#pragma once

#include <core/datatypes.h>

class ParticleVector;

class Forcing_ConstDP
{
private:
	float3 extraForce;

public:
	Forcing_ConstDP(float3 extraForce) : extraForce(extraForce) {}
	void setup(ParticleVector* pv, float t) {}

	__device__ __forceinline__ float3 operator()(float3 original, Particle p) const
	{
		return extraForce + original;
	}
};
