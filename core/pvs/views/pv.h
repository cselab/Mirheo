#pragma once

#include <core/utils/cuda_common.h>

/**
 * GPU-compatible struct of all the relevant data
 */
struct PVview
{
	float3 localDomainSize = {0,0,0};
	float3 globalDomainStart = {0,0,0};

	int size = 0;
	float4 *particles = nullptr;
	float4 *forces = nullptr;

	float mass = 0, invMass = 0;


	__forceinline__ __host__ __device__ float3 local2global(float3 x) const
	{
		return x + globalDomainStart + 0.5f * localDomainSize;
	}
	__forceinline__ __host__ __device__ float3 global2local(float3 x) const
	{
		return x - globalDomainStart - 0.5f * localDomainSize;
	}

	PVview(ParticleVector* pv = nullptr, LocalParticleVector* lpv = nullptr)
	{
		if (pv == nullptr || lpv == nullptr) return;

		localDomainSize = pv->localDomainSize;
		globalDomainStart = pv->globalDomainStart;

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


