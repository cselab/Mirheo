#include <core/object_vector.h>
#include <core/rigid_object_vector.h>
#include <core/helper_math.h>
#include <core/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce.h>

#pragma once

// https://arxiv.org/pdf/0811.2889.pdf
__device__ __forceinline__ float4 f3toQ(const float3 vec)
{
	return make_float4(0.0f, vec.x, vec.y, vec.z);
}
__device__ __forceinline__ float4 invQ(const float4 q)
{
	return make_float4(q.x, -q.y, -q.z, -q.w);
}

__device__ __forceinline__ float4 multiplyQ(const float4 q1, const float4 q2)
{
	float4 res;
	res.x =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
	res.y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
	res.z =  q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
	res.w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
	return res;
}

// rotate a point v in 3D space around the origin using this quaternion
__device__ __forceinline__ float3 rotate(const float3 x, const float4 q)
{
	float4 qX = make_float4(0.0f, x);
	qX = multiplyQ(multiplyQ(q, qX), invQ(q));

	return make_float3(qX.y, qX.z, qX.w);
}

__device__ __forceinline__ float4 compute_dq_dt(const float4 q, const float3 omega)
{
	return 0.5f*multiplyQ(f3toQ(omega), q);
}


__device__ inline float ellipsoidF(const float3 r, const float3 invAxes)
{
	return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1;
}

__global__ void bounceEllipsoid(float4* coosvels, float mass, const LocalObjectVector::COMandExtent* props, LocalRigidObjectVector::RigidMotion* motions,
		const int nObj, const float3 invAxes,
		const uint* __restrict__ cellsStartSize, CellListInfo cinfo, const float dt)
{
	const int objId = blockDim.x * blockIdx.x;
	if (objId >= nObj) return;

	const int3 cidLow  = cinfo.getCellIdAlongAxis(props[objId].low);
	const int3 cidHigh = cinfo.getCellIdAlongAxis(props[objId].high);
	const int3 span = cidHigh - cidLow + make_int3(1);
	const int totCells = span.x * span.y * span.z;

	auto motion = motions[objId];

	for (int i=threadIdx.x; i<totCells; i+=blockDim.x)
	{
		const int3 cid3 = make_int3( i/(span.y*span.x), (i/span.x) % span.y, i % span.x ) + cidLow;
		const int cid = cinfo.encode(cid3);

		int2 start_size = cinfo.decodeStartSize(cellsStartSize[cid]);

		for (int pid = start_size.x; pid < start_size.x + start_size.y; pid++)
		{
			const Particle p(coosvels[2*pid], coosvels[2*pid+1]);

			float3 coo = rotate(p.r - props  [objId].com, invQ(motion.q));
			float3 vel = rotate(p.u - motions[objId].vel, invQ(motion.q));

			float3 oldCoo = coo - vel*dt;


			auto F = [invAxes, motion, dt] (const float3 r) {
				return ellipsoidF(r, invAxes);
			};

			float alpha = bounceLinSearch(oldCoo, coo, F);

			if (alpha > -0.1f)
			{
				printf("%d: %f -> %f  %f  ==> ", p.i1, F(oldCoo), F(coo), alpha);

				coo =  oldCoo + (coo-oldCoo)*alpha;
				vel = -vel;

				printf("%f\n", F(coo));

				const float3 frc = 2.0f*mass*vel;
				motions[objId].force  += frc;
				motions[objId].torque += cross(coo, frc);

				coo = rotate(coo, motion.q) + props  [objId].com;
				vel = rotate(vel, motion.q) + motions[objId].vel;

				coosvels[2*pid]   = Float3_int(coo, p.i1).toFloat4();
				coosvels[2*pid+1] = Float3_int(vel, p.i2).toFloat4();
			}
		}
	}
}


__global__ void collectRigidForces(const float4 * coosvels, LocalRigidObjectVector::RigidMotion* motion, LocalRigidObjectVector::COMandExtent* props,
		const int nObj, const int objSize)
{
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	const int objId = gid >> 5;
	const int tid = gid & 0x1f;
	if (objId >= nObj) return;

	float3 force  = make_float3(0);
	float3 torque = make_float3(0);
	const float3 com = props[objId].com;

	// Find the total force and torque
#pragma unroll 3
	for (int i = tid; i < objSize; i += warpSize)
	{
		const int offset = (objId * objSize + i);

		const float3 frc = make_float3(coosvels[offset]);
		const float3 r   = make_float3(coosvels[offset*2]) - com;

		force += frc;
		torque += cross(r, frc);
	}

	force  = warpReduce( force,  [] (float a, float b) { return a+b; } );
	torque = warpReduce( torque, [] (float a, float b) { return a+b; } );

	if (tid == 0)
	{
		motion[objId].force  += force;
		motion[objId].torque += torque;
	}
}

/**
 * J is the diagonal moment of inertia tensor, J_1 is its inverse (simply 1/Jii)
 * Velocity-Verlet fused is used at the moment
 */
__global__ void integrateRigidMotion(LocalRigidObjectVector::RigidMotion* motions, const float3 J, const float3 J_1, const int nObj, const float dt)
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
	float3 dw_dt = J_1 * tau - J_1 * cross(omega, J*omega);
	float4 dq_dt = compute_dq_dt(q, omega);

	// d^2q / dt^2 = 1/2 * (dw/dt*q + w*dq/dt)
	float4 d2q_dt2 = 0.5f*(multiplyQ(f3toQ(dw_dt), q) + multiplyQ(f3toQ(omega), dq_dt));

	omega += dw_dt   * dt;
	dq_dt += d2q_dt2 * dt;
	q     += dq_dt   * dt;

	// Normalize q
	q = normalize(q);

	motions[objId].q      = q;
	motions[objId].omega  = omega;
	motions[objId].deltaQ = q - motions[objId].q;
	motions[objId].deltaV = dw_dt * dt;

	//**********************************************************************************
	// Translation
	//**********************************************************************************
	float3 force = motions[objId].force;
	float3 vel = motions[objId].vel;
	vel += force*dt;

	motions[objId].r     += vel*dt;
	motions[objId].vel    = vel;
	motions[objId].deltaW = force*dt;
	motions[objId].deltaR = vel*dt;
}

__global__ void applyRigidMotion(float4 * coosvels, LocalRigidObjectVector::RigidMotion* motions, const int nObj, const int objSize)
{
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;
	const int objId = pid % objSize;

	if (pid >= nObj*objSize) return;

	const auto motion = motions[objId];

	Particle p(coosvels[2*pid], coosvels[2*pid+1]);

	// Translation
	p.r += motion.deltaR;
	p.u += motion.deltaV;

	// Rotation
	p.r = rotate(p.r, motion.deltaQ);
	p.u = rotate(p.u, motion.deltaQ) + cross(motion.deltaW, p.r);

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









