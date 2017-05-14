#include <core/integrate.h>
#include <core/particle_vector.h>
#include <core/object_vector.h>
#include <core/logger.h>
#include <core/helper_math.h>
#include <core/cuda_common.h>

// Workaround for nsight
#ifndef __CUDACC_EXTENDED_LAMBDA__
#define __device__
#endif

/**
 * transform(float4& x, float4& v, const float4 f, const float invm, const float dt):
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
	float4 frc = forces[pid];

	// Send velocity to adjacent thread that has the coordinate
	float4 othval;
	othval.x = __shfl_down(val.x, 1);
	othval.y = __shfl_down(val.y, 1);
	othval.z = __shfl_down(val.z, 1);
	othval.w = __shfl_down(val.w, 1);

	// val is coordinate, othval is corresponding velocity
	if (sh == 0)
		transform(val, othval, frc, invmass, dt);

	// val is velocity, othval is rubbish
	if (sh == 1)
		transform(othval, val, frc, invmass, dt);

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

/**
 * Free flow
 */
void integrateNoFlow(ParticleVector* pv, const float dt, cudaStream_t stream)
{
	auto noflow = [] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt) {
		_noflow(x, v, f, invm, dt);
	};

	debug2("Integrating %d %s particles, timestep is %f", pv->np, pv->name.c_str(), dt);
	integrationKernel<<< (2*pv->np + 127)/128, 128, 0, stream >>>((float4*)pv->coosvels.devPtr(), (float4*)pv->forces.devPtr(), pv->np, 1.0/pv->mass, dt, noflow);
}

/**
 * Applied additional force to every particle
 */
void integrateConstDP(ParticleVector* pv, const float dt, cudaStream_t stream, float3 extraForce)
{
	auto constDP = [extraForce] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt) {
		_constDP(x, v, f, invm, dt, extraForce);
	};

	debug2("Integrating %d %s particles, timestep is %f", pv->np, pv->name.c_str(), dt);
	integrationKernel<<< (2*pv->np + 127)/128, 128, 0, stream >>>((float4*)pv->coosvels.devPtr(), (float4*)pv->forces.devPtr(), pv->np, 1.0/pv->mass, dt, constDP);
}

/**
 * Rotate with constant angular velocity omega around x0, regardless force
 */
void integrateConstOmega(ParticleVector* pv, const float dt, cudaStream_t stream, const float3 omega, const float3 x0)
{
	// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

	const float3 locX0 = x0 - pv->globalDomainStart;

	const float IomegaI = sqrt(dot(omega, omega));
	const float phi     = IomegaI * dt;
	const float sphi    = sin(phi);
	const float cphi    = cos(phi);

	const float3 k = omega / IomegaI;

	auto rotate = [k, sphi, cphi, locX0] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt) {
		float3 r = make_float3(x) - locX0;
		r = r * cphi + cross(k, r)*sphi * k*dot(k, r) * (1-cphi);
		x.x = r.x;
		x.y = r.y;
		x.z = r.z;
	};

	integrationKernel<<< (2*pv->np + 127)/128, 128, 0, stream >>>((float4*)pv->coosvels.devPtr(), (float4*)pv->forces.devPtr(), pv->np, 1.0/pv->mass, dt, rotate);
}

void integrateRigid(ObjectVector* ov, const float dt, cudaStream_t stream, float3 extraForce)
{
	auto noflow = [] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt) {
		_noflow(x, v, f, invm, dt);
	};

	debug2("Integrating %d objecst %s, timestep is %f", ov->nObjects, ov->name.c_str(), dt);

	const int nthreads = 128;
	integrateRigidKernel<<< (ov->nObjects*32 + nthreads-1)/nthreads, nthreads, 0, stream >>> (
			(float4*)ov->coosvels.devPtr(), (float4*)ov->forces.devPtr(), ov->com_extent.devPtr(),
			ov->nObjects, ov->objSize, 1.0/ov->mass, dt, noflow);
}










