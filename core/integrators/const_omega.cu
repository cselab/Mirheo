#include "const_omega.h"
#include "integration_kernel.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>

/**
 * Rotate with constant angular velocity omega around x0, regardless force
 */
void IntegratorConstOmega::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
	const float3 locX0 = pv->domain.global2local(center);

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

	PVview pvView(pv, pv->local());
	SAFE_KERNEL_LAUNCH(
			integrationKernel,
			getNblocks(2*pvView.size, nthreads), nthreads, 0, stream,
			pvView, dt, rotate );

	// PV may have changed, invalidate all
	pv->haloValid = false;
	pv->redistValid = false;
	pv->cellListStamp++;
}
