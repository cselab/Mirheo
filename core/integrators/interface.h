#pragma once

#include <string>

class ParticleVector;

struct Integrator
{
	std::string name;
	float dt;

	virtual void stage1(ParticleVector* pv, cudaStream_t stream) = 0;
	virtual void stage2(ParticleVector* pv, cudaStream_t stream) = 0;

	Integrator(std::string name, float dt) : dt(dt), name(name) {}

	virtual ~Integrator() = default;
};
