#pragma once

#include "interface.h"

struct IntegratorOscillate : Integrator
{
	float3 vel;
	int count{0};
	int period;

	void stage1(ParticleVector* pv, float t, cudaStream_t stream) override {};
	void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

	IntegratorOscillate(std::string name, float dt, float3 vel, int period);

	~IntegratorOscillate() = default;
};
