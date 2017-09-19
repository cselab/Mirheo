#pragma once

#include "interface.h"

struct IntegratorVVConstDP : Integrator
{
	float3 extraForce;

	void stage1(ParticleVector* pv, cudaStream_t stream) override;
	void stage2(ParticleVector* pv, cudaStream_t stream) override;

	IntegratorVVConstDP(std::string name, float dt, float3 extraForce) :
		Integrator(name, dt),
		extraForce(extraForce)
	{}

	~IntegratorVVConstDP() = default;
};
