#pragma once

#include <core/object_vector.h>
#include <core/rigid_object_vector.h>
#include <core/cuda_common.h>
#include <core/rigid_kernels/quaternion.h>


__global__ void collectRigidForces(const float4 * coosvels, const float4 * forces,
		LocalRigidObjectVector::RigidMotion* motion, LocalRigidObjectVector::COMandExtent* props,
		const int nObj, const int objSize)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= nObj) return;

	float3 force  = make_float3(0);
	float3 torque = make_float3(0);
	const float3 com = props[objId].com;

	// Find the total force and torque
#pragma unroll 3
	for (int i = tid; i < objSize; i += blockDim.x)
	{
		const int offset = (objId * objSize + i);

		const float3 frc = make_float3(forces[offset]);
		const float3 r   = make_float3(coosvels[offset*2]) - com;

		force += frc;
		torque += cross(r, frc);
	}

	force  = warpReduce( force,  [] (float a, float b) { return a+b; } );
	torque = warpReduce( torque, [] (float a, float b) { return a+b; } );

	if (tid % warpSize == 0)
	{
		atomicAdd(&motion[objId].force,  force);
		atomicAdd(&motion[objId].torque, torque);
	}
}

/**
 * J is the diagonal moment of inertia tensor, J_1 is its inverse (simply 1/Jii)
 * Velocity-Verlet fused is used at the moment
 */
__global__ void integrateRigidMotion(LocalRigidObjectVector::RigidMotion* motions,
		const float3 J, const float3 J_1, const float invMass, const int nObj, const float dt)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= nObj) return;

	//**********************************************************************************
	// Rotation
	//**********************************************************************************
	float4 q     = motions[objId].q;
	float3 omega = motions[objId].omega;
	float3 tau   = motions[objId].torque;

	// tau = J dw/dt + w x Jw  =>  dw/dt = J'*tau - J'*(w x Jw)
	float3 dw_dt = J_1 * tau ;//- J_1 * cross(omega, J*omega);
	omega += dw_dt * dt;

	// XXX: using OLD q and NEW w ?
	// d^2q / dt^2 = 1/2 * (dw/dt*q + w*dq/dt)
	float4 dq_dt = compute_dq_dt(q, omega);
	float4 d2q_dt2 = 0.5f*(multiplyQ(f3toQ(dw_dt), q) + multiplyQ(f3toQ(omega), dq_dt));

	dq_dt += d2q_dt2 * dt;
	q     += dq_dt   * dt;

	// Normalize q
	q = normalize(q);

	motions[objId].deltaQ = q - motions[objId].q;
	motions[objId].deltaW = dw_dt * dt;
	motions[objId].q      = q;
	motions[objId].omega  = omega;

	//**********************************************************************************
	// Translation
	//**********************************************************************************
	float3 force = motions[objId].force;
	float3 vel = motions[objId].vel;
	vel += force*dt * invMass;

	motions[objId].r     += vel*dt;
	motions[objId].vel    = vel;
	motions[objId].deltaV = force*dt * invMass;
	motions[objId].deltaR = vel*dt;


//	printf("obj  %d  f [%f %f %f],  t [%f %f %f],  r [%f %f %f]   v [%f %f %f] \n"
//				"    q [%f %f %f %f]   w [%f %f %f],  dq [%f %f %f %f] \n", objId,
//				motions[objId].force.x,  motions[objId].force.y,  motions[objId].force.z,
//				motions[objId].torque.x, motions[objId].torque.y, motions[objId].torque.z,
//				motions[objId].r.x,  motions[objId].r.y,  motions[objId].r.z,
//				motions[objId].vel.x,  motions[objId].vel.y,  motions[objId].vel.z,
//				motions[objId].q.x,  motions[objId].q.y,  motions[objId].q.z, motions[objId].q.w,
//				motions[objId].omega.x,  motions[objId].omega.y,  motions[objId].omega.z,
//				motions[objId].deltaQ.x,  motions[objId].deltaQ.y,  motions[objId].deltaQ.z, motions[objId].deltaQ.w);
//
//		printf("    dr [%f %f %f]   dv [%f %f %f]   dw [%f %f %f]\n\n",
//				motions[objId].deltaR.x,  motions[objId].deltaR.y,  motions[objId].deltaR.z,
//				motions[objId].deltaV.x,  motions[objId].deltaV.y,  motions[objId].deltaV.z,
//				motions[objId].deltaW.x,  motions[objId].deltaW.y,  motions[objId].deltaW.z);
}


// TODO: rotate initial config instead of incremental rotations
__global__ void applyRigidMotion(float4 * coosvels, LocalRigidObjectVector::RigidMotion* motions, const int nObj, const int objSize)
{
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;
	const int objId = pid / objSize;

	if (pid >= nObj*objSize) return;

	const auto motion = motions[objId];

	const Particle p_orig(coosvels[2*pid], coosvels[2*pid+1]);
	Particle p = p_orig;

	// Translation
	p.r += motion.deltaR;

	// Rotation
	float4 dq = multiplyQ(motion.q, invQ(motion.q - motion.deltaQ));

	p.r = rotate(p.r - motion.r, dq) + motion.r;
	p.u = motion.vel + cross(motion.omega, p.r - motion.r);
//
//	if (p.s21 == 42)
//	{
//		float3 tmp = rotate(p.r - motion.r, invQ(motion.q));
//		printf("rotatatataing %d :  %f %f %f\n", p.s21,  tmp.x, tmp.y, tmp.z);
//	}


	coosvels[2*pid]   = make_float4(p.r, __int_as_float(p.i1));
	coosvels[2*pid+1] = make_float4(p.u, __int_as_float(p.i2));
}

__global__ void clearRigidForces(LocalRigidObjectVector::RigidMotion* motions, const int nObj)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= nObj) return;

	motions[objId].force  = make_float3(0.0f);
	motions[objId].torque = make_float3(0.0f);
}



