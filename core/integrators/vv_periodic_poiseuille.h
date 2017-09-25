#pragma once

#include "interface.h"

struct IntegratorVVPeriodicPoiseuille : Integrator
{
	float force;
	enum class Direction {x, y, z} dir;
	float3 globalDomainSize;

	void stage1(ParticleVector* pv, cudaStream_t stream) override;
	void stage2(ParticleVector* pv, cudaStream_t stream) override;

	IntegratorVVPeriodicPoiseuille(std::string name, float dt, float force, Direction dir, float3 globalDomainSize) :
		Integrator(name, dt),
		force(force), dir(dir), globalDomainSize(globalDomainSize)
	{}

	~IntegratorVVPeriodicPoiseuille() = default;
};
