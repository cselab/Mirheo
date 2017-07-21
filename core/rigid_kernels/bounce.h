#pragma once

#include <core/rigid_object_vector.h>
#include <core/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce_solver.h>
#include <core/rigid_kernels/quaternion.h>

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

		if (alpha > -0.1f)
		{
			float3 bounced = oldCoo + (coo-oldCoo)*alpha;

			// Change velocity's frame to the object frame, correct for rotation as well
			float3 vel = rotate(p.u - motions[objId].vel, invQ(objQ));
			vel -= cross(motions[objId].omega, bounced);
			vel = -vel;
			const float3 frc = 2.0f*mass*vel;
			vel += cross(motions[objId].omega, bounced);

			atomicAdd(&motions[objId].force, frc);
			atomicAdd(&motions[objId].torque, cross(coo, frc));

			if (F(bounced) < 0.0f)
				printf("Fail on particle %d\n", p.i1);

			// Return to the original frame
			bounced = rotate(bounced, objQ) + objR;
			vel 	= rotate(vel,     objQ) + motions[objId].vel;

			coosvels[2*pid]   = Float3_int(bounced, p.i1).toFloat4();
			coosvels[2*pid+1] = Float3_int(vel,     p.i2).toFloat4();
		}
		else
			printf("Fail on particle %d\n", p.i1);
	}
}

__launch_bounds__(128, 7)
__global__ void bounceEllipsoid(float4* coosvels, float mass,
		const LocalRigidObjectVector::COMandExtent* props, LocalRigidObjectVector::RigidMotion* motions,
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
				int id = atomicAggInc((int*)&nCells);
				validCells[id] = cinfo.encode(cid3);
			}
		}

		__syncthreads();

		// If we have enough cells ready - process them
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

	// Process remaining
	bounceCellArray(validCells, nCells, objId, coosvels,
						mass, motions, invAxes,
						cellsStartSize, cinfo, dt);
}



