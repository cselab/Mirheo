#pragma once

#include <core/utils/cuda_common.h>

/**
 * GPU-compatible struct of all the relevant data
 */
struct PVview
{
	int size = 0;
	float4 *particles = nullptr;
	float4 *forces = nullptr;

	float mass = 0, invMass = 0;

	PVview(ParticleVector* pv = nullptr, LocalParticleVector* lpv = nullptr)
	{
		if (pv == nullptr || lpv == nullptr) return;

		size = lpv->size();
		particles = reinterpret_cast<float4*>(lpv->coosvels.devPtr());
		forces    = reinterpret_cast<float4*>(lpv->forces.devPtr());

		mass = pv->mass;
		invMass = 1.0 / mass;
	}
};


struct PVviewWithOldParticles : PVview
{
	float4 *oldParticles = nullptr;

	PVviewWithOldParticles(ParticleVector* pv = nullptr, LocalParticleVector* lpv = nullptr) :
		PVview(pv, lpv)
	{
		// Setup extra fields
		if (lpv != nullptr)
			oldParticles = lpv->getDataPerParticle<float4>("old_particles")->devPtr();
	}
};


