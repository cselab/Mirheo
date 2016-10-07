#pragma once

#include "containers.h"
#include "non_cached_rw.h"

// combine with rearrange
__global__ void integrateKernel(float4* xyzouvwo, const float4* axayaz, const int n, const float dt)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 copies coordinates, sh = 1 -- velocity
	if (pid >= n) return;


	const float4 a = axayaz  [pid];
	float4 val     = xyzouvwo[gid];

	if (sh == 1) // val is velocity here
		val += a*dt;

	float4 updVel;
	updVel.x = __shfl_down(val.x, 1);
	updVel.y = __shfl_down(val.y, 1);
	updVel.z = __shfl_down(val.z, 1);
	//updVel.w = __shfl_down(val.w, 1);

	if (sh == 0) // val is coordinate here
		val += updVel*dt;

	xyzouvwo[gid] = val;
}


void integrate(ParticleVector& pv, const float dt, cudaStream_t stream)
{
	integrateKernel<<< (2*pv.np + 127)/128, 128, 0, stream >>>((float4*)pv.coosvels.devdata, (float4*)pv.accs.devdata, pv.np, dt);
}
