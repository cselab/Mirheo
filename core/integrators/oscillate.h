#pragma once

#include "interface.h"

/**
 * Apply periodic sine wave to the particle velocities.
 * Coordinate is computed by Velocity-Verlet scheme (same as
 * Euler in this case)
 */
class IntegratorOscillate : public Integrator
{
public:

	void stage1(ParticleVector* pv, float t, cudaStream_t stream) override {};
	void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

	IntegratorOscillate(std::string name, float dt, float3 vel, float period);

	~IntegratorOscillate() = default;

private:

	float3 vel;    ///< Velocity amplitude
	float period;  ///< Sine wave period
};
