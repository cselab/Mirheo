#pragma once

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>

class ParticleVector;

class Forcing_PeriodicPoiseuille
{
public:
	enum class Direction {x, y, z};

	Forcing_PeriodicPoiseuille(float magnitude, Direction dir) :
		magnitude(magnitude)
	{
		switch (dir)
		{
			case Direction::x: _dir = 0; break;
			case Direction::y: _dir = 1; break;
			case Direction::z: _dir = 2; break;
		}
	}

	void setup(ParticleVector* pv, float t)
	{
		domain = pv->domain;
	}

	__device__ inline float3 operator()(float3 original, Particle p) const
	{
		float3 gr = domain.local2global(p.r);
		float3 ef{0.0f,0.0f,0.0f};

		if (_dir == 0) ef.x = gr.y > 0.5f*domain.globalSize.y ? magnitude : -magnitude;
		if (_dir == 1) ef.y = gr.z > 0.5f*domain.globalSize.z ? magnitude : -magnitude;
		if (_dir == 2) ef.z = gr.x > 0.5f*domain.globalSize.x ? magnitude : -magnitude;

		return ef + original;
	}

private:
	float magnitude;
	int _dir;

	DomainInfo domain;
};
