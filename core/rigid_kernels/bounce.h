#pragma once

#include <core/pvs/rigid_object_vector.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce_solver.h>
#include <core/rigid_kernels/quaternion.h>

__device__ inline float ellipsoidF(const float3 r, const float3 invAxes)
{
	return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1.0f;
}

__device__ __forceinline__ void bounceCellArray(
		REOVview ovView, PVview pvView, int objId,
		int* validCells, int nCells,
		CellListInfo cinfo, const float dt)
{
	const float threshold = 2e-6f;
	const int maxIters = 100;

	auto motions = ovView.motions;
	float3 axes = ovView.axes;
	float3 invAxes = ovView.invAxes;


	if (threadIdx.x >= nCells) return;

	float3 objR = motions[objId].r;
	float4 objQ = motions[objId].q;

	// Prepare rolling back in time
	float3 oldR = objR - motions[objId].vel * dt;
	float4 oldQ = motions[objId].prevQ;
	oldQ = normalize(oldQ);

	int cid = validCells[threadIdx.x];
	int pstart = cinfo.cellStarts[cid];
	int pend = cinfo.cellStarts[cid+1];

	// XXX: changing reading layout may improve performance here
	for (int pid = pstart; pid < pend; pid++)
	{
		const Particle p(pvView.particles, pid);

		// Go to the obj frame where the obj is completely still
		float3 coo = rotate(p.r - objR, invQ(objQ));

		// For the old coordinate use the motion description of past timestep
		float3 oldCoo = p.r - p.u*dt;
		oldCoo = rotate(oldCoo - oldR, invQ(oldQ));
		float3 dr = coo - oldCoo;

		float vold = ellipsoidF(oldCoo, invAxes);
		float vcur = ellipsoidF(coo,    invAxes);

		// If the particle is outside - skip it, it's fine
		if (vcur >= 0.0f) continue;

		// Correct the old position so that it is outside
		// Inside may happen because of imprecise arithmetics
		if (vold <= 0.0f)
		{
			float3 normal = normalize(make_float3(
					axes.y*axes.y * axes.z*axes.z * oldCoo.x,
					axes.z*axes.z * axes.x*axes.x * oldCoo.y,
					axes.x*axes.x * axes.y*axes.y * oldCoo.z));

			for (int i=0; i<maxIters; i++)
			{
				oldCoo += 5*threshold*normal;
				vold = ellipsoidF(oldCoo, invAxes);
				if (vold > 0.0f) break;
			}
		}

		float alpha = solveLinSearch( [=] (const float lambda) { return ellipsoidF(oldCoo + (coo-oldCoo)*lambda, invAxes);} );
		float3 newCoo = oldCoo + (coo-oldCoo)*alpha;

		// Push out a little bit
		float3 normal = normalize(make_float3(
				axes.y*axes.y * axes.z*axes.z * newCoo.x,
				axes.z*axes.z * axes.x*axes.x * newCoo.y,
				axes.x*axes.x * axes.y*axes.y * newCoo.z));

		for (int i=0; i<maxIters; i++)
		{
			newCoo += 5*threshold*normal;
			if (ellipsoidF(newCoo, invAxes) > 0.0f) break;
		}

		if (ellipsoidF(newCoo, invAxes) < 0.0f)
		{
			printf("Bounce-back failed on particle %d (%f %f %f)  %f -> %f to %f, alpha %f. Recovering to old position\n",
					p.i1, p.r.x, p.r.y, p.r.z,
					ellipsoidF(oldCoo, invAxes), ellipsoidF(coo, invAxes),
					ellipsoidF(newCoo, invAxes), alpha);

			newCoo = oldCoo;
		}

		// Change velocity's frame to the object frame, correct for rotation as well
		float3 vel = rotate(p.u - motions[objId].vel, invQ(objQ));
		vel -= cross(motions[objId].omega, newCoo);
		vel = -vel;
		vel += cross(motions[objId].omega, newCoo);

		// Return to the original frame
		newCoo = rotate(newCoo, objQ) + objR;
		vel    = rotate(vel,    objQ) + motions[objId].vel;

		const float3 frc = -pvView.mass * (vel - p.u) / dt;
		atomicAdd( &motions[objId].force,  frc);
		atomicAdd( &motions[objId].torque, cross(newCoo - objR, frc) );

		pvView.particles[2*pid]   = Float3_int(newCoo, p.i1).toFloat4();
		pvView.particles[2*pid+1] = Float3_int(vel,    p.i2).toFloat4();
	}
}

__global__ void bounceEllipsoid(REOVview ovView, PVview pvView,
		CellListInfo cinfo, const float dt)
{
	const float threshold = 0.2f;

	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= ovView.nObjects) return;

	// Preparation step. Filter out all the cells that don't intersect the surface
	__shared__ volatile int nCells;
	extern __shared__ int validCells[];

	nCells = 0;
	__syncthreads();

	const int3 cidLow  = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].low  - 0.5f);
	const int3 cidHigh = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].high + 1.5f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	const float4 invq = invQ(ovView.motions[objId].q);

	for (int i=tid; i<totCells + blockDim.x-1; i+=blockDim.x)
	{
		if (i < totCells)
		{
			const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;

			float3 v000 = make_float3(cid3) * cinfo.h - cinfo.localDomainSize*0.5f - ovView.motions[objId].r;

			float3 v001 = rotate( v000 + make_float3(        0,         0, cinfo.h.z), invq );
			float3 v010 = rotate( v000 + make_float3(        0, cinfo.h.y,         0), invq );
			float3 v011 = rotate( v000 + make_float3(        0, cinfo.h.y, cinfo.h.z), invq );
			float3 v100 = rotate( v000 + make_float3(cinfo.h.x,         0,         0), invq );
			float3 v101 = rotate( v000 + make_float3(cinfo.h.x,         0, cinfo.h.z), invq );
			float3 v110 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y,         0), invq );
			float3 v111 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y, cinfo.h.z), invq );

			v000 = rotate( v000, invq );

			if ( ellipsoidF(v000, ovView.invAxes) < threshold ||
				 ellipsoidF(v001, ovView.invAxes) < threshold ||
				 ellipsoidF(v010, ovView.invAxes) < threshold ||
				 ellipsoidF(v011, ovView.invAxes) < threshold ||
				 ellipsoidF(v100, ovView.invAxes) < threshold ||
				 ellipsoidF(v101, ovView.invAxes) < threshold ||
				 ellipsoidF(v110, ovView.invAxes) < threshold ||
				 ellipsoidF(v111, ovView.invAxes) < threshold )
			{
				int cid = cinfo.encode(cid3);
				if (cid < cinfo.totcells)
				{
					int id = atomicAggInc((int*)&nCells);
					validCells[id] = cid;
				}
			}
		}

		__syncthreads();

		// If we have enough cells ready - process them
		if (nCells >= blockDim.x)
		{
			bounceCellArray(ovView, pvView, objId, validCells, blockDim.x, cinfo, dt);

			__syncthreads();

			if (tid == 0) nCells -= blockDim.x;
			validCells[tid] = validCells[tid + blockDim.x];

			__syncthreads();
		}
	}

	__syncthreads();

	// Process remaining
	bounceCellArray(ovView, pvView, objId, validCells, nCells, cinfo, dt);
}



