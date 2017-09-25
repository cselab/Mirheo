#include "vv_periodic_poiseuille.h"
#include "integration_kernel.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>


void IntegratorVVPeriodicPoiseuille::stage1(ParticleVector* pv, cudaStream_t stream)
{ }

void IntegratorVVPeriodicPoiseuille::stage2(ParticleVector* pv, cudaStream_t stream)
{
	PVinfo pvinfo = pv->pvInfo();

	auto st2 = [*this, pvinfo] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
		float3 gr = pvinfo.local2global(p.r);
		float3 ef{0,0,0};

		if (dir == Direction::x)
			ef.x = gr.y > 0.5*globalDomainSize.y ? force : -force;

		if (dir == Direction::y)
			ef.y = gr.z > 0.5*globalDomainSize.z ? force : -force;

		if (dir == Direction::z)
			ef.z = gr.x > 0.5*globalDomainSize.x ? force : -force;

		p.u += (f+ef)*invm*dt;
		p.r += p.u*dt;
	};

	int nthreads = 128;
	debug2("Integrating (stage 2) %d %s particles with periodic poiseuille force %f, timestep is %f",
			force, pv->local()->size(), pv->name.c_str(), dt);

	if (pv->local()->size() > 0)
		integrationKernel<<< getNblocks(2*pv->local()->size(), nthreads), nthreads, 0, stream >>>(
				(float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, st2);
	pv->local()->changedStamp++;
}
