#include "integrate.h"
#include "containers.h"

template<typename Transform>
__global__ void integrationKernel(float4* coosvels, const float4* forces, const int n, const float dt, const float invmass, Transform transform)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 loads coordinate, sh = 1 -- velocity
	if (pid >= n) return;

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

void integrateNoFlow(ParticleVector* pv, const float dt, cudaStream_t stream)
{
	auto noflow = [] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt, const int pid) {
		_noflow(x, v, f, invm, dt);
	};

	debug2("Integrating %d %s particles, timestep is %f", pv->np, pv->name.c_str(), dt);
	integrationKernel<<< (2*pv->np + 127)/128, 128, 0, stream >>>((float4*)pv->coosvels.devPtr(), (float4*)pv->forces.devPtr(), pv->np, dt, 1.0/pv->mass, noflow);
}

void integrateConstDP(ParticleVector* pv, const float dt, cudaStream_t stream, float3 extraForce)
{
	auto constDP = [extraForce] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt, const int pid) {
		_constDP(x, v, f, invm, dt, extraForce);
	};

	debug2("Integrating %d %s particles, timestep is %f", pv->np, pv->name.c_str(), dt);
	integrationKernel<<< (2*pv->np + 127)/128, 128, 0, stream >>>((float4*)pv->coosvels.devPtr(), (float4*)pv->forces.devPtr(), pv->np, dt, 1.0/pv->mass, constDP);
}









