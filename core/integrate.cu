#include <core/integrate.h>
#include <core/particle_vector.h>
#include <core/object_vector.h>
#include <core/logger.h>
#include <core/helper_math.h>
#include <core/cuda_common.h>
#include <core/rigid_kernels.h>

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

	debug2("Integrating %d %s particles, timestep is %f", pv->local()->size(), pv->name.c_str(), dt);
	integrationKernel<<< (2*pv->local()->size() + 127)/128, 128, 0, stream >>>((float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, noflow);
}

/**
 * Applied additional force to every particle
 */
void integrateConstDP(ParticleVector* pv, const float dt, cudaStream_t stream, float3 extraForce)
{
	auto constDP = [extraForce] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt) {
		_constDP(x, v, f, invm, dt, extraForce);
	};

	debug2("Integrating %d %s particles with extra force [%8.5f %8.5f %8.5f], timestep is %f",
			pv->local()->size(), pv->name.c_str(), extraForce.x, extraForce.y, extraForce.z, dt);
	integrationKernel<<< getNblocks(2*pv->local()->size(), 128), 128, 0, stream >>>((float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, constDP);
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

	integrationKernel<<< getNblocks(2*pv->local()->size(), 128), 128, 0, stream >>>((float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, rotate);
}

/**
 * Assume that the forces are not yet distributed
 * Also integrate object's Q
 * Only VV integration now
 */
void integrateRigid(RigidObjectVector* ov, const float dt, cudaStream_t stream, float3 extraForce)
{
	debug2("Integrating %d rigid objects %s, timestep is %f", ov->local()->nObjects, ov->name.c_str(), dt);

	collectRigidForces<<< getNblocks(2*ov->local()->size(), 128), 128, 0, stream >>>
			((float4*)ov->local()->coosvels.devPtr(), ov->local()->motions.devPtr(), ov->local()->comAndExtents.devPtr(), ov->local()->nObjects, ov->local()->objSize);

	auto sq = [] (float x) { return x*x; };
	const float3 J = 5.0/ov->mass * make_float3(
			1.0/(sq(ov->axes.y) + sq(ov->axes.z)),
			1.0/(sq(ov->axes.z) + sq(ov->axes.x)),
			1.0/(sq(ov->axes.x) + sq(ov->axes.y)) );
	const float3 J_1 = 1.0 / J;

	integrateRigidMotion<<< getNblocks(ov->nObjects, 64), 64, 0, stream >>>(ov->local()->motions.devPtr(), J, J_1, ov->nObjects, dt);

	applyRigidMotion<<< getNblocks(ov->local()->size(), 128), 128, 0, stream >>>((float4*)ov->local()->coosvels.devPtr(), ov->local()->motions.devPtr(), ov->nObjects, ov->objSize);

	clearRigidForces<<< getNblocks(ov->nObjects, 64), 64, 0, stream >>>(ov->local()->motions.devPtr(), ov->nObjects);
}










