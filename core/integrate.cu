#include <core/integrate.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/logger.h>
#include <core/helper_math.h>
#include <core/cuda_common.h>
#include <core/rigid_kernels/integration.h>

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
		p = Particle(othval, val);
		transform(p, frc.v, invmass, dt);
		val = p.u2Float4();
	}

	coosvels[gid] = val; //writeNoCache(coosvels + gid, val);
}

//==============================================================================================
//==============================================================================================

/**
 * Free flow
 */
void IntegratorVVNoFlow::stage1(ParticleVector* pv, cudaStream_t stream)
{
//	auto st1 = [] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
//		p.u += 0.5*f*invm*dt;
//		p.r += p.u*dt;
//	};
//
//	int nthreads = 128;
//	debug2("Integrating (stage 1) %d %s particles, timestep is %f", pv->local()->size(), pv->name.c_str(), dt);
//	integrationKernel<<< getNblocks(2*pv->local()->size(), nthreads), nthreads, 0, stream >>>(
//			(float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, st1);
//	pv->local()->changedStamp++;
}

void IntegratorVVNoFlow::stage2(ParticleVector* pv, cudaStream_t stream)
{
	auto st2 = [] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
		p.u += f*invm*dt;
		p.r += p.u*dt;
	};

	int nthreads = 128;
	debug2("Integrating (stage 2) %d %s particles, timestep is %f", pv->local()->size(), pv->name.c_str(), dt);
	
	if (pv->local()->size() > 0)
		integrationKernel<<< getNblocks(2*pv->local()->size(), nthreads), nthreads, 0, stream >>>(
				(float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, st2);
	pv->local()->changedStamp++;
}

/**
 * Applied additional force to every particle
 */
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
	auto ef = extraForce;
	auto st2 = [ef] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
		p.u += (f+ef)*invm*dt;
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

/**
 * Rotate with constant angular velocity omega around x0, regardless force
 */
void IntegratorConstOmega::stage2(ParticleVector* pv, cudaStream_t stream)
{
	// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

	const float3 locX0 = center - pv->globalDomainStart;

	const float IomegaI = sqrt(dot(omega, omega));
	const float phi     = IomegaI * dt;
	const float sphi    = sin(phi);
	const float cphi    = cos(phi);

	const float3 k = omega / IomegaI;

	auto rotate = [k, sphi, cphi, locX0] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
		float3 r = p.r - locX0;
		r = r * cphi + cross(k, r)*sphi * k*dot(k, r) * (1-cphi);
		p.r = r + locX0;
	};

	int nthreads = 128;
	
	if (pv->local()->size() > 0)
		integrationKernel<<< getNblocks(2*pv->local()->size(), nthreads), nthreads, 0, stream >>>(
				(float4*)pv->local()->coosvels.devPtr(), (float4*)pv->local()->forces.devPtr(), pv->local()->size(), 1.0/pv->mass, dt, rotate);

	pv->local()->changedStamp++;
}

/**
 * Assume that the forces are not yet distributed
 * Also integrate object's Q
 * Only VV integration now
 */
// FIXME: split VV into two stages
void IntegratorVVRigid::stage2(ParticleVector* pv, cudaStream_t stream)
{
	RigidEllipsoidObjectVector* ov = dynamic_cast<RigidEllipsoidObjectVector*> (pv);
	if (ov == nullptr) die("Rigid integration only works with rigid objects, can't work with %s", pv->name.c_str());
	debug("Integrating %d rigid objects %s (total %d particles), timestep is %f", ov->local()->nObjects, ov->name.c_str(), ov->local()->size(), dt);
	if (ov->local()->nObjects == 0)
		return;

	collectRigidForces<<< getNblocks(2*ov->local()->size(), 128), 128, 0, stream >>> (
			(float4*)ov->local()->coosvels.devPtr(), (float4*)ov->local()->forces.devPtr(), ov->local()->motions.devPtr(),
			ov->local()->comAndExtents.devPtr(), ov->local()->nObjects, ov->local()->objSize);

	const float3 J = ov->objMass / 5.0 * make_float3(
			sqr(ov->axes.y) + sqr(ov->axes.z),
			sqr(ov->axes.z) + sqr(ov->axes.x),
			sqr(ov->axes.x) + sqr(ov->axes.y) );

	const float3 J_1 = 1.0 / J;

	integrateRigidMotion<<< getNblocks(ov->local()->nObjects, 64), 64, 0, stream >>>(ov->local()->motions.devPtr(), J, J_1, 1.0 / ov->objMass, ov->local()->nObjects, dt);

	applyRigidMotion<<< getNblocks(ov->local()->size(), 128), 128, 0, stream >>>(
			(float4*)ov->local()->coosvels.devPtr(), ov->initialPositions.devPtr(), ov->local()->motions.devPtr(), ov->local()->nObjects, ov->objSize);

	clearRigidForces<<< getNblocks(ov->local()->nObjects, 64), 64, 0, stream >>>(ov->local()->motions.devPtr(), ov->local()->nObjects);

	pv->local()->changedStamp++;
}



