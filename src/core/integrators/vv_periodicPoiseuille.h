#pragma once

#include "interface.h"
#include <memory>

/**
 * Implementation of Velocity-Verlet integration in one step
 */
struct IntegratorVV_periodicPoiseuille : Integrator
{
	std::unique_ptr<Integrator> impl;

	void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
	void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

	IntegratorVV_periodicPoiseuille(std::string name, float dt, float force, std::string direction);

	~IntegratorVV_periodicPoiseuille();
};

