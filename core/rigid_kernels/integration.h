#pragma once

#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/utils/cuda_common.h>
#include <core/rigid_kernels/quaternion.h>

/**
 * Find total force and torque on objects, write it to motions
 */
static __global__ void collectRigidForces(ROVview ovView)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= ovView.nObjects) return;

	RigidReal3 force {0,0,0};
	RigidReal3 torque{0,0,0};
	float3 com = make_float3( ovView.motions[objId].r );

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

	force  = warpReduce( force,  [] (RigidReal a, RigidReal b) { return a+b; } );
	torque = warpReduce( torque, [] (RigidReal a, RigidReal b) { return a+b; } );

	if ( (tid % warpSize) == 0 )
	{
		atomicAdd(&ovView.motions[objId].force,  force);
		atomicAdd(&ovView.motions[objId].torque, torque);
	}
}

/**
 * J is the diagonal moment of inertia tensor, J_1 is its inverse (simply 1/Jii)
 * Velocity-Verlet fused is used at the moment
 */
static __global__ void integrateRigidMotion(ROVview_withOldMotion ovView, const float dt)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= ovView.nObjects) return;

	auto& motion = ovView.motions[objId];
	ovView.old_motions[objId] = motion;

	//**********************************************************************************
	// Rotation
	//**********************************************************************************
	auto q = motion.q;

	// Update angular velocity in the body frame
	auto omega = rotate(motion.omega,  invQ(q));
	auto tau   = rotate(motion.torque, invQ(q));

	// tau = J dw/dt + w x Jw  =>  dw/dt = J'*tau - J'*(w x Jw)
	// J is the diagonal inertia tensor in the body frame
	auto dw_dt = ovView.J_1 * (tau - cross(omega, ovView.J*omega));
	omega += dw_dt * dt;

	// Only for output purposes
	auto L = rotate(omega*ovView.J, motion.q);

	omega = rotate(omega, motion.q);

	// using OLD q and NEW w ?
	// d^2q / dt^2 = 1/2 * (dw/dt*q + w*dq/dt)
	auto dq_dt = compute_dq_dt(q, omega);
	auto d2q_dt2 = 0.5f*(multiplyQ(f3toQ(dw_dt), q) + multiplyQ(f3toQ(omega), dq_dt));

	dq_dt += d2q_dt2 * dt;
	q     += dq_dt   * dt;

	// Normalize q
	q = normalize(q);

	motion.omega = omega;
	motion.q     = q;

	//**********************************************************************************
	// Translation
	//**********************************************************************************
	auto force = motion.force;
	auto vel   = motion.vel;
	vel += force*dt * ovView.invObjMass;

	motion.vel = vel;
	motion.r += vel*dt;
//
//	printf("obj  %d  r [%f %f %f]   v [%f %f %f],  f [%f %f %f],  t [%f %f %f],   \n"
//			"    q [%f %f %f %f]   w [%f %f %f]   L [%f %f %f] \n", ovView.ids[objId],
//			motion.r.x,  motion.r.y,  motion.r.z,
//			motion.vel.x,  motion.vel.y,  motion.vel.z,
//			motion.force.x,  motion.force.y,  motion.force.z,
//			motion.torque.x, motion.torque.y, motion.torque.z ,
//			motion.q.x,  motion.q.y,  motion.q.z, motion.q.w,
//			motion.omega.x,  motion.omega.y,  motion.omega.z,
//			L.x, L.y, L.z);
}


/**
 * Rotates and translates the initial sample according to new position and orientation
 */
static __global__ void applyRigidMotion(ROVview ovView, const float4 * __restrict__ initial)
{
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;
	const int objId = pid / ovView.objSize;
	const int locId = pid % ovView.objSize;

	if (pid >= ovView.nObjects*ovView.objSize) return;

	const auto motion = toSingleMotion(ovView.motions[objId]);

	Particle p(ovView.particles, pid);

	// Some explicit conversions for double precision
	p.r = motion.r + rotate( f4tof3(initial[locId]), motion.q );
	p.u = motion.vel + cross(motion.omega, p.r - motion.r);

	ovView.particles[2*pid]   = p.r2Float4();
	ovView.particles[2*pid+1] = p.u2Float4();
}

static __global__ void clearRigidForces(ROVview ovView)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= ovView.nObjects) return;

	ovView.motions[objId].force  = {0,0,0};
	ovView.motions[objId].torque = {0,0,0};
}



