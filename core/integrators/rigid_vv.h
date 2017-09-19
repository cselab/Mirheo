#pragma once

#include "interface.h"

struct IntegratorVVRigid : Integrator
{
	void stage1(ParticleVector* pv, cudaStream_t stream) override;
	void stage2(ParticleVector* pv, cudaStream_t stream) override;

	IntegratorVVRigid(std::string name, float dt) :
		Integrator(name, dt)
	{}

	~IntegratorVVRigid() = default;
};
