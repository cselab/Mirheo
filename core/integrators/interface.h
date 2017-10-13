#pragma once

#include <string>

class ParticleVector;

class Integrator
{
public:
	std::string name;
	float dt;

	virtual void stage1(ParticleVector* pv, float t, cudaStream_t stream) = 0;
	virtual void stage2(ParticleVector* pv, float t, cudaStream_t stream) = 0;

	Integrator(std::string name, float dt) : dt(dt), name(name) {}

	virtual ~Integrator() = default;
};
