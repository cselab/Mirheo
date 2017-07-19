#include <core/object_vector.h>
#include <core/rigid_object_vector.h>
#include <core/helper_math.h>
#include <core/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce.h>

#pragma once

// http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
// https://arxiv.org/pdf/0811.2889.pdf
__device__ __host__ __forceinline__ float4 f3toQ(const float3 vec)
{
	return make_float4(0.0f, vec.x, vec.y, vec.z);
}
__device__ __host__ __forceinline__ float4 invQ(const float4 q)
{
	return make_float4(q.x, -q.y, -q.z, -q.w);
}

__device__ __host__ __forceinline__ float4 multiplyQ(const float4 q1, const float4 q2)
{
	float4 res;
	res.x =  q1.x * q2.x - q1.y * q2.y - q1.z * q2.z - q1.w * q2.w;
	res.y =  q1.x * q2.y + q1.y * q2.x + q1.z * q2.w - q1.w * q2.z;
	res.z =  q1.x * q2.z - q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
	res.w =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
	return res;
}

// rotate a point v in 3D space around the origin using this quaternion
__device__ __host__ __forceinline__ float3 rotate(const float3 x, const float4 q)
{
	float4 qX = make_float4(0.0f, x);
	qX = multiplyQ(multiplyQ(q, qX), invQ(q));

	return make_float3(qX.y, qX.z, qX.w);
}

__device__ __host__ __forceinline__ float4 compute_dq_dt(const float4 q, const float3 omega)
{
	return 0.5f*multiplyQ(f3toQ(omega), q);
}


__device__ inline float ellipsoidF(const float3 r, const float3 invAxes)
{
	return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1;
}


__device__ __forceinline__ void bounceCellArray(int* validCells, int nCells, int objId, float4* coosvels,
		float mass, LocalRigidObjectVector::RigidMotion* motions, const float3 invAxes,
		const uint* __restrict__ cellsStartSize, CellListInfo cinfo, const float dt)
{
	if (threadIdx.x >= nCells) return;

	float3 objR = motions[objId].r;
	float4 objQ = motions[objId].q;

	// Prepare rolling back in time
	float3 oldR = objR - motions[objId].vel * dt;
	float4 oldQ = objQ - compute_dq_dt(objQ, motions[objId].omega) * dt;
	oldQ = normalize(oldQ);

	auto F = [invAxes, dt] (const float3 r) {
		return ellipsoidF(r, invAxes) - 5e-6f;
	};

	int2 start_size = cinfo.decodeStartSize(cellsStartSize[validCells[threadIdx.x]]);

	// XXX: changing reading layout may improve performance here
	for (int pid = start_size.x; pid < start_size.x + start_size.y; pid++)
	{
		const Particle p(coosvels[2*pid], coosvels[2*pid+1]);

		// Go to the obj frame where the obj is completely still
		float3 coo = rotate(p.r - objR, invQ(objQ));
		if (F(coo) > 0.0f) continue;

		// For the old coordinate use the motion description of past timestep
		float3 oldCoo = p.r - p.u*dt;
		oldCoo = rotate(oldCoo - oldR, invQ(oldQ));

		float alpha = bounceLinSearch(oldCoo, coo, F);

//			if (p.i1 == 472323)
//				printf("%8d obj  %d  [%f %f %f] -> [%f %f %f] -rot-> [%f %f %f]  (%f)\n"
//						"   oldCoo: [%f %f %f] -> [%f %f %f] -rot-> [%f %f %f] (%f)\n\n",
//						p.i1, objId, p.r.x, p.r.y, p.r.z,
//						p.r.x - objR.x, p.r.y - objR.y, p.r.z - objR.z,
//						coo.x, coo.y, coo.z, F(coo),
//						p.r.x - p.u.x*dt, p.r.y - p.u.y*dt, p.r.z - p.u.z*dt,
//						p.r.x - p.u.x*dt - oldR.x, p.r.y - p.u.y*dt - oldR.y, p.r.z - p.u.z*dt - oldR.z,
//						oldCoo.x, oldCoo.y, oldCoo.z, F(oldCoo) + 5e-5);

		if (alpha > -0.1f)
		{
			float3 bounced = oldCoo + (coo-oldCoo)*alpha;

			float3 vel = rotate(p.u - motions[objId].vel, invQ(objQ));
			vel -= cross(motions[objId].omega, bounced);
			vel = -vel;
			vel += cross(motions[objId].omega, bounced);

//				printf("%d: %f -> %f  (%f)  ==> %f\n", p.i1, F(oldCoo) + 5e-6, F(coo) + 5e-6, alpha, F(bounced));

			const float3 frc = 2.0f*mass*vel;
			motions[objId].force  += frc;
			motions[objId].torque += cross(coo, frc);

			bounced = rotate(bounced, objQ) + objR;
			vel 	= rotate(vel,     objQ) + motions[objId].vel;

			coosvels[2*pid]   = Float3_int(bounced, p.i1).toFloat4();
			coosvels[2*pid+1] = Float3_int(vel,     p.i2).toFloat4();
		}
	}
}

__launch_bounds__(128, 7)
__global__ void bounceEllipsoid(float4* coosvels, float mass, const LocalObjectVector::COMandExtent* props, LocalRigidObjectVector::RigidMotion* motions,
		const int nObj, const float3 invAxes,
		const uint* __restrict__ cellsStartSize, CellListInfo cinfo, const float dt)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= nObj) return;

	// Preparation step. Filter out all the cells that don't intersect the surface
	__shared__ volatile int nCells;
	__shared__ int validCells[256];

	nCells = 0;
	__syncthreads();

	const int3 cidLow  = cinfo.getCellIdAlongAxis(props[objId].low  - make_float3(0.3f));
	const int3 cidHigh = cinfo.getCellIdAlongAxis(props[objId].high + make_float3(0.3f));
	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	const float4 invq = invQ(motions[objId].q);

	for (int i=tid; i<totCells + blockDim.x-1; i+=blockDim.x)
	{
		if (i < totCells)
		{
			const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;

			float3 v000 = make_float3(cid3) * cinfo.h - cinfo.domainSize*0.5f - motions[objId].r;

			float3 v001 = rotate( v000 + make_float3(        0,         0, cinfo.h.z), invq );
			float3 v010 = rotate( v000 + make_float3(        0, cinfo.h.y,         0), invq );
			float3 v011 = rotate( v000 + make_float3(        0, cinfo.h.y, cinfo.h.z), invq );
			float3 v100 = rotate( v000 + make_float3(cinfo.h.x,         0,         0), invq );
			float3 v101 = rotate( v000 + make_float3(cinfo.h.x,         0, cinfo.h.z), invq );
			float3 v110 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y,         0), invq );
			float3 v111 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y, cinfo.h.z), invq );

			v000 = rotate( v000, invq );

			if ( ellipsoidF(v000, invAxes) < 0.0f ||
				 ellipsoidF(v001, invAxes) < 0.0f ||
				 ellipsoidF(v010, invAxes) < 0.0f ||
				 ellipsoidF(v011, invAxes) < 0.0f ||
				 ellipsoidF(v100, invAxes) < 0.0f ||
				 ellipsoidF(v101, invAxes) < 0.0f ||
				 ellipsoidF(v110, invAxes) < 0.0f ||
				 ellipsoidF(v111, invAxes) < 0.0f )
			{
				int id = atomicAdd((int*)&nCells, 1);
				validCells[id] = cinfo.encode(cid3);
			}
		}

		__syncthreads();

		if (nCells >= blockDim.x)
		{
			bounceCellArray(validCells, blockDim.x, objId, coosvels,
					mass, motions, invAxes,
					cellsStartSize, cinfo, dt);

			__syncthreads();

			if (tid == 0) nCells -= blockDim.x;
			validCells[tid] = validCells[tid + blockDim.x];

			__syncthreads();
		}
	}

	__syncthreads();

	bounceCellArray(validCells, nCells, objId, coosvels,
						mass, motions, invAxes,
						cellsStartSize, cinfo, dt);
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









