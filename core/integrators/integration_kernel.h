#pragma once

#include <core/utils/cuda_common.h>
#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>

/**
 * transform(Particle&p, const float3 f, const float invm, const float dt):
 *  performs integration
 *
 * Will read from .old_particles and write to .particles
 */
template<typename Transform>
__global__ void integrationKernel(PVview_withOldParticles pvView, const float dt, Transform transform)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 loads coordinate, sh = 1 -- velocity
	if (pid >= pvView.size) return;

	float4 val = readNoCache(pvView.old_particles + gid);
	Float3_int frc(pvView.forces[pid]);

	// Send velocity to adjacent thread that has the coordinate
	Particle p;
	float4 othval;
	othval.x = __shfl_down(val.x, 1);
	othval.y = __shfl_down(val.y, 1);
	othval.z = __shfl_down(val.z, 1);
	othval.w = __shfl_down(val.w, 1);

	// val is coordinate, othval is corresponding velocity
	if (sh == 0)
	{
		p = Particle(val, othval);
		transform(p, frc.v, pvView.invMass, dt);
		val = p.r2Float4();
	}

	// val is velocity, othval is rubbish
	if (sh == 1)
	{
		// to distinguish this case
		othval.w = __int_as_float(-1);

		p = Particle(othval, val);
		transform(p, frc.v, pvView.invMass, dt);
		val = p.u2Float4();
	}

	writeNoCache(pvView.particles + gid, val);
}
