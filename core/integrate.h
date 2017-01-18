#pragma once

template<typename Transform>
__global__ void integrationKernel(float4* coosvels, const float4* forces, const float invmass, const int n, const float dt, Transform transform)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 loads coordinate, sh = 1 -- velocity
	if (pid >= n) return;

	// instead of:
	// const float4 val = in_xyzouvwo[gid];
	//
	// this is to allow more cache for atomics
	// loads / stores here need no cache
	float4 val = coosvels[gid]; //readNoCache(coosvels+gid);
	float4 frc = forces[pid];


	// Send velocity to adjacent thread that has the coordinate
	float4 othval;
	othval.x = __shfl_down(val.x, 1);
	othval.y = __shfl_down(val.y, 1);
	othval.z = __shfl_down(val.z, 1);
	othval.w = __shfl_down(val.w, 1);

	// val is coordinate, othval is corresponding velocity
	if (sh == 0)
		transform(val, othval, frc, invmass, dt, pid);

	// val is velocity, othval is rubbish
	if (sh == 1)
		transform(othval, val, frc, invmass, dt, pid);

	coosvels[gid] = val; //writeNoCache(coosvels + gid, val);
}

//==============================================================================================
//==============================================================================================

__device__ __forceinline__ void _noflow (float4& x, float4& v, const float4 f, const float invm, const float dt)
{
	v.x += f.x*invm*dt;
	v.y += f.y*invm*dt;
	v.z += f.z*invm*dt;

	x.x += v.x*dt;
	x.y += v.y*dt;
	x.z += v.z*dt;
}

__device__ __forceinline__ void _constDP (float4& x, float4& v, const float4 f, const float invm, const float dt, const float3 extraForce)
{
	v.x += (f.x+extraForce.x) * invm*dt;
	v.y += (f.y+extraForce.y) * invm*dt;
	v.z += (f.z+extraForce.z) * invm*dt;

	x.x += v.x*dt;
	x.y += v.y*dt;
	x.z += v.z*dt;
}
