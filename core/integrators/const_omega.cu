#include "const_omega.h"
#include "integration_kernel.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>

/**
 * Rotate with constant angular velocity omega around x0, regardless force
 */
void IntegratorConstOmega::stage2(ParticleVector* pv, cudaStream_t stream)
{
	const float3 locX0 = center - pv->globalDomainStart;

	const float IomegaI = sqrt(dot(omega, omega));
	const float phi     = IomegaI * dt;
	const float sphi    = sin(phi);
	const float cphi    = cos(phi);

	const float3 k = omega / IomegaI;

	// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
	auto rotate = [k, sphi, cphi, locX0] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
		float3 r = p.r - locX0;
		r = r * cphi + cross(k, r)*sphi * k*dot(k, r) * (1-cphi);
		p.r = r + locX0;
	};

	int nthreads = 128;

	if (pv->local()->size() > 0)
	{
		auto pvView = create_PVview(pv, pv->local());
		integrationKernel<<< getNblocks(2*pvView.size, nthreads), nthreads, 0, stream >>>(pvView, dt, rotate);
	}

	pv->local()->changedStamp++;
}
