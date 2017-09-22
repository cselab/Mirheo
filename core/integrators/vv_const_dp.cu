#include "vv_const_dp.h"
#include "integration_kernel.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>


void IntegratorVVConstDP::stage1(ParticleVector* pv, cudaStream_t stream)
{
//	auto st1 = [=] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
//		p.u += 0.5*(f+extraForce)*invm*dt;
//		p.r += p.u*dt;
//	};
//
//	int nthreads = 128;
//	debug2("Integrating (stage 1) %d %s particles with extra force [%8.5f %8.5f %8.5f], timestep is %f", pv->local()->size(), pv->name.c_str(), dt);
//	integrationKernel<<< getNblocks(2*pv->local()->size(), nthreads), nthreads, 0, stream >>>(
//			(float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, st1);
//	pv->local()->changedStamp++;
}

void IntegratorVVConstDP::stage2(ParticleVector* pv, cudaStream_t stream)
{
	auto st2 = [*this] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
		p.u += (f+extraForce)*invm*dt;
		p.r += p.u*dt;
	};

	int nthreads = 128;
	debug2("Integrating (stage 2) %d %s particles with extra force [%8.5f %8.5f %8.5f], timestep is %f",
			pv->local()->size(), pv->name.c_str(), extraForce.x, extraForce.y, extraForce.z, dt);

	if (pv->local()->size() > 0)
		integrationKernel<<< getNblocks(2*pv->local()->size(), nthreads), nthreads, 0, stream >>>(
				(float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, st2);
	pv->local()->changedStamp++;
}
