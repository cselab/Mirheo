#pragma once

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
};

static PVview create_PVview(ParticleVector* pv, LocalParticleVector* lpv)
{
	PVview view;
	if (pv == nullptr || lpv == nullptr)
		return view;

	view.localDomainSize = pv->localDomainSize;
	view.globalDomainStart = pv->globalDomainStart;

	view.size = lpv->size();
	view.particles = reinterpret_cast<float4*>(lpv->coosvels.devPtr());
	view.forces    = reinterpret_cast<float4*>(lpv->forces.devPtr());

	view.mass = pv->mass;
	view.invMass = 1.0 / view.mass;

	return view;
}


struct PVviewWithOldParticles : PVview
{
	float4 *oldParticles = nullptr;
};

static PVview create_PVviewWithOldParticles(ParticleVector* pv, LocalParticleVector* lpv)
{
	// Create a default view
	PVviewWithOldParticles view;
	view.PVview::operator= ( create_PVview(pv, lpv) );

	// Setup extra fields

	view.oldParticles = lpv->getDataPerParticle<float4>("old_particles")->devPtr();
	return view;
}

