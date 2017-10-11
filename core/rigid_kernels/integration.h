#pragma once

#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/utils/cuda_common.h>
#include <core/rigid_kernels/quaternion.h>


__global__ void collectRigidForces(ROVview ovView)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= ovView.nObjects) return;

	float3 force  = make_float3(0);
	float3 torque = make_float3(0);
	const float3 com = ovView.motions[objId].r;

	// Find the total force and torque
#pragma unroll 3
	for (int i = tid; i < ovView.objSize; i += blockDim.x)
	{
		const int offset = (objId * ovView.objSize + i);

		const float3 frc = make_float3(ovView.forces[offset]);
		const float3 r   = make_float3(ovView.particles[offset*2]) - com;

		force += frc;
		torque += cross(r, frc);
	}

	force  = warpReduce( force,  [] (float a, float b) { return a+b; } );
	torque = warpReduce( torque, [] (float a, float b) { return a+b; } );

	if ( (tid % warpSize) == 0)
	{
		atomicAdd(&ovView.motions[objId].force,  force);
		atomicAdd(&ovView.motions[objId].torque, torque);
	}
}

/**
 * J is the diagonal moment of inertia tensor, J_1 is its inverse (simply 1/Jii)
 * Velocity-Verlet fused is used at the moment
 */
__global__ void integrateRigidMotion(ROVview ovView, const float dt)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= ovView.nObjects) return;

	auto motions = ovView.motions;

	//**********************************************************************************
	// Rotation
	//**********************************************************************************
	float4 q     = motions[objId].q;
	float3 omega = motions[objId].omega;
	float3 tau   = motions[objId].torque;

	// FIXME allow for non-diagonal inertia tensors

	// tau = J dw/dt + w x Jw  =>  dw/dt = J'*tau - J'*(w x Jw)
	float3 dw_dt = ovView.J_1 * tau - ovView.J_1 * cross(omega, ovView.J*omega);
	omega += dw_dt * dt;

	// XXX: using OLD q and NEW w ?
	// d^2q / dt^2 = 1/2 * (dw/dt*q + w*dq/dt)
	float4 dq_dt = compute_dq_dt(q, omega);
	float4 d2q_dt2 = 0.5f*(multiplyQ(f3toQ(dw_dt), q) + multiplyQ(f3toQ(omega), dq_dt));

	dq_dt += d2q_dt2 * dt;
	q     += dq_dt   * dt;

	// Normalize q
	q = normalize(q);

	motions[objId].prevQ  = motions[objId].q;
	motions[objId].q      = q;
	motions[objId].omega  = omega;

	//**********************************************************************************
	// Translation
	//**********************************************************************************
	float3 force = motions[objId].force;
	float3 vel = motions[objId].vel;
	vel += force*dt * ovView.invObjMass;

	motions[objId].r += vel*dt;
	motions[objId].vel = vel;
//
//	printf("obj  %d  r [%f %f %f]   v [%f %f %f],  f [%f %f %f],  t [%f %f %f],   \n"
//			/*"    q [%f %f %f %f]   w [%f %f %f],  ooldq [%f %f %f %f] \n"*/, ovView.ids[objId],
//			motions[objId].r.x,  motions[objId].r.y,  motions[objId].r.z,
//			motions[objId].vel.x,  motions[objId].vel.y,  motions[objId].vel.z,
//			motions[objId].force.x,  motions[objId].force.y,  motions[objId].force.z,
//			motions[objId].torque.x, motions[objId].torque.y, motions[objId].torque.z /*,
//			motions[objId].q.x,  motions[objId].q.y,  motions[objId].q.z, motions[objId].q.w,
//			motions[objId].omega.x,  motions[objId].omega.y,  motions[objId].omega.z,
//			motions[objId].prevQ.x,  motions[objId].prevQ.y,  motions[objId].prevQ.z, motions[objId].prevQ.w */);
}


// TODO: rotate initial config instead of incremental rotations
__global__ void applyRigidMotion(ROVview ovView, const float4 * __restrict__ initial)
{
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;
	const int objId = pid / ovView.objSize;
	const int locId = pid % ovView.objSize;

	if (pid >= ovView.nObjects*ovView.objSize) return;

	const auto motion = ovView.motions[objId];

	Particle p(ovView.particles, pid);

	p.r = motion.r + rotate( f4tof3(initial[locId]), motion.q );
	p.u = motion.vel + cross(motion.omega, p.r - motion.r);

	ovView.particles[2*pid]   = p.r2Float4();
	ovView.particles[2*pid+1] = p.u2Float4();
}

__global__ void clearRigidForces(ROVview ovView)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= ovView.nObjects) return;

	ovView.motions[objId].force  = make_float3(0.0f);
	ovView.motions[objId].torque = make_float3(0.0f);
}



