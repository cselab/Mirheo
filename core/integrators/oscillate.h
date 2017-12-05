#pragma once

#include "interface.h"

/**
 * Apply periodic sine wave to the particle velocities.
 * Coordinate is computed by Velocity-Verlet scheme (same as
 * Euler in this case)
 *
 * \rst
 * .. attention::
 *    In current implementation only works correctly is applied
 *    to no more than one ParticleVector
 * \endrst
 */
class IntegratorOscillate : Integrator
{
public:

	void stage1(ParticleVector* pv, float t, cudaStream_t stream) override {};
	void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

	IntegratorOscillate(std::string name, float dt, float3 vel, int period);

	~IntegratorOscillate() = default;

private:

	float3 vel;   ///< Velocity amplitude
	int period;   ///< Sine wave period
	int count{0}; ///< Internal counter
};
