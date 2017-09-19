#pragma once

#include "interface.h"

struct IntegratorVV : Integrator
{
	void stage1(ParticleVector* pv, cudaStream_t stream) override;
	void stage2(ParticleVector* pv, cudaStream_t stream) override;

	IntegratorVV(std::string name, float dt) :
		Integrator(name, dt)
	{}

	~IntegratorVV() = default;
};
