#pragma once

#include <core/cuda_common.h>
#include <core/datatypes.h>


/**
 * transform(Particle&p, const float3 f, const float invm, const float dt):
 *  performs integration
 */
template<typename Transform>
__global__ void integrationKernel(float4* coosvels, const float4* forces, const int n, const float invmass, const float dt, Transform transform)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 loads coordinate, sh = 1 -- velocity
	if (pid >= n) return;

	float4 val = coosvels[gid]; //readNoCache(coosvels+gid);
	Float3_int frc(forces[pid]);

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
		transform(p, frc.v, invmass, dt);
		val = p.r2Float4();
	}

	// val is velocity, othval is rubbish
	if (sh == 1)
	{
		// to distinguish this case
		othval.w = __int_as_float(-1);

		p = Particle(othval, val);
		transform(p, frc.v, invmass, dt);
		val = p.u2Float4();
	}

	coosvels[gid] = val; //writeNoCache(coosvels + gid, val);
}
