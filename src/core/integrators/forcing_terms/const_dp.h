#pragma once

#include <core/datatypes.h>
#include <core/utils/cuda_common.h>

class ParticleVector;

/**
 * Applies constant force #extraForce to every particle
 */
class Forcing_ConstDP
{
public:
	Forcing_ConstDP(float3 extraForce) : extraForce(extraForce) {}
	void setup(ParticleVector* pv, float t) {}

	__device__ inline float3 operator()(float3 original, Particle p) const
	{
		return extraForce + original;
	}

private:
	float3 extraForce;
};
