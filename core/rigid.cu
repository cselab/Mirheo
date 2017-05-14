#include <core/object_vector.h>
#include <core/rigid_object_vector.h>
#include <core/helper_math.h>
#include <core/cuda_common.h>

// https://arxiv.org/pdf/0811.2889.pdf
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

__device__ __forceinline__ float4 dq_dt(const float4 q, const float3 omega)
{
	const float4 w = make_float4(0, omega.x, omega.y, omega.z);
	return 0.5f*multiplyQ(w, q);
}


__device__ inline float ellipse(const float3 r, const float3 invAxes)
{
	return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1;
}

__global__ void bounceEllipsoid(float4* coosvels, const ObjectVector::COMandExtent* props, RigidObjectVector::RigidMotion* motions, const int nObj, const float3 invAxes, const float r,
		const uint* __restrict__ cellsStartSize, CellListInfo cinfo, const float dt)
{
	const int objId = blockDim.x * blockIdx.x;
	if (objId >= nObj) return;

	const int3 cidLow  = cinfo.getCellIdAlongAxis(props[objId].low);
	const int3 cidHigh = cinfo.getCellIdAlongAxis(props[objId].high);
	const int3 span = cidHigh - cidLow;
	const int totCells = span.x * span.y * span.z;

	auto motion = motions[objId];

	for (int i=threadIdx.x; i<totCells; i+=blockDim.x)
	{
		const int3 cid3 = make_int3( i/(span.y*span.x), (i/span.x) % span.y, i % span.x ) + cidLow;
		const int cid = cinfo.encode(cid3);

		int2 start_size = cellsStartSize[cid];

		for (int pid = start_size.x; pid < start_size.x + start_size.y; pid++)
		{
			float3 coo = make_float3(coosvels[2*pid]);
			float3 vel = make_float3(coosvels[2*pid+1]);

			coo -= props[objId].com;
			vel -= motions[objId].vel;

			float4 invInstQ_2 = invQ(dq_dt(motion.q, 0.5f*motion.omega));
			coo = rotate(coo, invQ(motion.q));

			auto F = [invAxes, motion] (const float3 r) {
				const float3 r_rot = rotate(coo, motion.q);
				return ellipse(r_rot, invAxes);
			};

			float alpha = bounceLinSearch(coo, vel, dt, F);


		}
	}
}


__global__ void collectRigidForces(const float4 * coosvels, RigidObjectVector::RigidMotion* motion, RigidObjectVector::COMandExtent* props, const int nObj, const int objSize)
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

	force.x  = __shfl(force.x, 0);
	force.y  = __shfl(force.y, 0);
	force.z  = __shfl(force.z, 0);

	torque.x = __shfl(torque.x, 0);
	torque.y = __shfl(torque.y, 0);
	torque.z = __shfl(torque.z, 0);

	if (tid == 0)
	{
		motion[objId].force = force;
		motion[objId].torque = torque;
	}
}

template<typename Transform>
__global__ void distributeRigidForces(const float4 * coosvels, float4* forces, RigidObjectVector::RigidMovement* movement, RigidObjectVector::COMandExtent* props, const int nObj, const int objSize)
{
	// http://math.stackexchange.com/questions/519200/dot-product-and-cross-product-solving-a-set-of-simultaneous-vector-equations
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	const int objId = gid >> 5;
	const int tid = gid & 0x1f;
	if (objId >= nObj) return;

	const float3 force =  movement[objId].force;
	const float3 torque = movement[objId].torque;

	// Distribute the force and torque per particle
#pragma unroll 3
	for (int i = tid; i < objSize; i += warpSize)
	{
		const int offset = (objId * objSize + i) * 2;

		float4 r = coosvels[offset];
		float4 v = coosvels[offset+1];

		// Force consists of translational and rotational components
		// first is just average force, second comes from a solution of:
		//
		//  torque = r x f,  f*r = 0
		//
		const float3 f = force + cross(torque, make_float3(r)) / dot(r, r);
		forces[offset] = make_float4(f);
	}
}
