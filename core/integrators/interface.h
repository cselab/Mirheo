#pragma once

#include <string>

class ParticleVector;

/**
 * Integrate ParticleVectors
 *
 * Should implement movement of the particles or objects due to the applied forces
 */
class Integrator
{
public:
	std::string name;
	float dt;

	virtual void stage1(ParticleVector* pv, float t, cudaStream_t stream) = 0;
	virtual void stage2(ParticleVector* pv, float t, cudaStream_t stream) = 0;

	/**
	 * Ask ParticleVectors which the class will be working with to have specific properties
	 * Default: ask nothing
	 * Called from Simulation right after setup
	 */
	virtual void setPrerequisites(ParticleVector* pv) {}

	Integrator(std::string name, float dt) : dt(dt), name(name) {}

	virtual ~Integrator() = default;
};
