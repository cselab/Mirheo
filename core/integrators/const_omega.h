#pragma once

#include "interface.h"

struct IntegratorConstOmega : Integrator
{
	float3 center, omega;

	void stage1(ParticleVector* pv, cudaStream_t stream) override {};
	void stage2(ParticleVector* pv, cudaStream_t stream) override;

	IntegratorConstOmega(std::string name, float dt, float3 center, float3 omega) :
		Integrator(name, dt),
		center(center),	omega(omega)
	{}

	~IntegratorConstOmega() = default;
};
