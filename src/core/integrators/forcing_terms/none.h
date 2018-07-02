#pragma once

#include <core/datatypes.h>

class ParticleVector;

class Forcing_None
{
public:
	Forcing_None() {}
	void setup(ParticleVector* pv, float t) {}

	__device__ inline float3 operator()(float3 original, Particle p) const
	{
		return original;
	}
};
