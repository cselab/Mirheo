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
		REOVview_withOldMotion ovView, PVview_withOldParticles pvView,
		int objId,
		int* validCells, int nCells,
		CellListInfo cinfo, const float dt)
{
	const float threshold = 1e-5f;

	auto motion     = ovView.motions[objId];
	auto old_motion = ovView.old_motions[objId];

	const float3 axes    = ovView.axes;
	const float3 invAxes = ovView.invAxes;

	if (threadIdx.x >= nCells) return;

	int cid = validCells[threadIdx.x];
	int pstart = cinfo.cellStarts[cid];
	int pend   = cinfo.cellStarts[cid+1];

	// XXX: changing reading layout may improve performance here
	for (int pid = pstart; pid < pend; pid++)
	{
		Particle p    (pvView.particles,     pid);
		Particle old_p(pvView.old_particles, pid);

		// Go to the obj frame of reference
		float3 coo    = rotate(p.r     - motion.r,     invQ(motion.q));
		float3 oldCoo = rotate(old_p.r - old_motion.r, invQ(old_motion.q));
		float3 dr = coo - oldCoo;

		// If the particle is outside - skip it, it's fine
		if (ellipsoidF(coo, invAxes) > 0.0f) continue;

		// This is intersection point
		float alpha = solveLinSearch( [=] (const float lambda) { return ellipsoidF(oldCoo + dr*lambda, invAxes);} );
		float3 newCoo = oldCoo + dr*min(alpha, 0.0f);

		// Push out a little bit
		float3 normal = normalize(make_float3(
				axes.y*axes.y * axes.z*axes.z * newCoo.x,
				axes.z*axes.z * axes.x*axes.x * newCoo.y,
				axes.x*axes.x * axes.y*axes.y * newCoo.z));

		newCoo += threshold*normal;

		// If smth went notoriously bad
		if (ellipsoidF(newCoo, invAxes) < 0.0f)
		{
			printf("Bounce-back failed on particle %d (%f %f %f)  %f -> %f to %f, alpha %f. Recovering to old position\n",
					p.i1, p.r.x, p.r.y, p.r.z,
					ellipsoidF(oldCoo, invAxes), ellipsoidF(coo, invAxes),
					ellipsoidF(newCoo, invAxes), alpha);

			newCoo = oldCoo;
		}

		// Return to the original frame
		newCoo = rotate(newCoo, motion.q) + motion.r;

		// Change velocity's frame to the object frame, correct for rotation as well
		float3 vEll = motion.vel + cross( motion.omega, newCoo-motion.r );
		float3 newU = vEll - (p.u - vEll);

		const float3 frc = -pvView.mass * (newU - p.u) / dt;
		atomicAdd( &ovView.motions[objId].force,  frc);
		atomicAdd( &ovView.motions[objId].torque, cross(newCoo - motion.r, frc) );

		p.r = newCoo;
		p.u = newU;
		p.write2Float4(pvView.particles, pid);
	}
}

__device__ __forceinline__ bool isValidCell(int3 cid3, LocalRigidObjectVector::RigidMotion motion, CellListInfo cinfo, float3 invAxes)
{
	const float threshold = 0.5f;

	float3 v000 = make_float3(cid3) * cinfo.h - cinfo.localDomainSize*0.5f - motion.r;
	const float4 invq = invQ(motion.q);

	float3 v001 = rotate( v000 + make_float3(        0,         0, cinfo.h.z), invq );
	float3 v010 = rotate( v000 + make_float3(        0, cinfo.h.y,         0), invq );
	float3 v011 = rotate( v000 + make_float3(        0, cinfo.h.y, cinfo.h.z), invq );
	float3 v100 = rotate( v000 + make_float3(cinfo.h.x,         0,         0), invq );
	float3 v101 = rotate( v000 + make_float3(cinfo.h.x,         0, cinfo.h.z), invq );
	float3 v110 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y,         0), invq );
	float3 v111 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y, cinfo.h.z), invq );

	v000 = rotate( v000, invq );

	return ( ellipsoidF(v000, invAxes) < threshold ||
			 ellipsoidF(v001, invAxes) < threshold ||
			 ellipsoidF(v010, invAxes) < threshold ||
			 ellipsoidF(v011, invAxes) < threshold ||
			 ellipsoidF(v100, invAxes) < threshold ||
			 ellipsoidF(v101, invAxes) < threshold ||
			 ellipsoidF(v110, invAxes) < threshold ||
			 ellipsoidF(v111, invAxes) < threshold );
}

__global__ void bounceEllipsoid(REOVview_withOldMotion ovView, PVview_withOldParticles pvView,
		CellListInfo cinfo, const float dt)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= ovView.nObjects) return;

	// Preparation step. Filter out all the cells that don't intersect the surface
	__shared__ volatile int nCells;
	extern __shared__ int validCells[];

	nCells = 0;
	__syncthreads();

	const int3 cidLow  = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].low  - 1.5f);
	const int3 cidHigh = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].high + 2.5f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

//	auto motion = ovView.motions[objId];
//	if(threadIdx.x == 0)
//	printf("obj  %d  r [%f %f %f]   v [%f %f %f],  f [%f %f %f],  t [%f %f %f],   \n"
//			"    q [%f %f %f %f]   w [%f %f %f] \n", ovView.ids[objId],
//			motion.r.x,  motion.r.y,  motion.r.z,
//			motion.vel.x,  motion.vel.y,  motion.vel.z,
//			motion.force.x,  motion.force.y,  motion.force.z,
//			motion.torque.x, motion.torque.y, motion.torque.z ,
//			motion.q.x,  motion.q.y,  motion.q.z, motion.q.w,
//			motion.omega.x,  motion.omega.y,  motion.omega.z);

	for (int i=tid; i<totCells + blockDim.x-1; i+=blockDim.x)
	{
		const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
		const int cid = cinfo.encode(cid3);

		if ( i < totCells &&
			 cid < cinfo.totcells &&
			 isValidCell(cid3, ovView.motions[objId], cinfo, ovView.invAxes) )
		{
			int id = atomicAggInc((int*)&nCells);
			validCells[id] = cid;
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


//=============================================================================================
// http://jamesgregson.blogspot.ch/2012/09/2x2-and-3x3-matrix-inverses-and-linear.html

__device__ __forceinline__ void invert_3x3( const float A[3][3], float iA[3][3] )
{
	double det;

	det =     A[0][0]*(A[2][2]*A[1][1]-A[2][1]*A[1][2])
			- A[1][0]*(A[2][2]*A[0][1]-A[2][1]*A[0][2])
			+ A[2][0]*(A[1][2]*A[0][1]-A[1][1]*A[0][2]);

	double inv_det = 1.0f / det;

	iA[0][0] =  (A[2][2]*A[1][1]-A[2][1]*A[1][2]) * inv_det;
	iA[0][1] = -(A[2][2]*A[0][1]-A[2][1]*A[0][2]) * inv_det;
	iA[0][2] =  (A[1][2]*A[0][1]-A[1][1]*A[0][2]) * inv_det;

	iA[1][0] = -(A[2][2]*A[1][0]-A[2][0]*A[1][2]) * inv_det;
	iA[1][1] =  (A[2][2]*A[0][0]-A[2][0]*A[0][2]) * inv_det;
	iA[1][2] = -(A[1][2]*A[0][0]-A[1][0]*A[0][2]) * inv_det;

	iA[2][0] =  (A[2][1]*A[1][0]-A[2][0]*A[1][1]) * inv_det;
	iA[2][1] = -(A[2][1]*A[0][0]-A[2][0]*A[0][1]) * inv_det;
	iA[2][2] =  (A[1][1]*A[0][0]-A[1][0]*A[0][1]) * inv_det;
}

__device__ __forceinline__ void solve_3x3( const float A[3][3], float x[3], const float b[3] )
{
	float iA[3][3];

	invert_3x3( A, iA );

	x[0] = (double)iA[0][0]*b[0] + iA[0][1]*b[1] + iA[0][2]*b[2];
	x[1] = (double)iA[1][0]*b[0] + iA[1][1]*b[1] + iA[1][2]*b[2];
	x[2] = (double)iA[2][0]*b[0] + iA[2][1]*b[1] + iA[2][2]*b[2];
}

__device__ __forceinline__ void skewCoordinates(float3 v, float3 e[3], float coo[3])
{
	// Metric tensor
	float g[3][3];

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			g[i][j] = dot(e[i], e[j]);

	// Right hand side
	float b[3];

	for (int i=0; i<3; i++)
		b[i] = dot(v, e[i]);

	solve_3x3(g, coo, b);
}

/**
 * Transfers force and torque onto two arbitrary particles
 * Total force and torque are preserved
 */
__global__ void rigidMotion2forces(ROVview ovView)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= ovView.nObjects) return;

	assert(ovView.objSize >= 3);

	const auto motion = ovView.motions[objId];

	if (dot(motion.torque, motion.torque) < 1e-5)
	{
		atomicAdd( ovView.forces + objId*ovView.objSize, motion.force );
		return;
	}

	float3 vs[3], ns[3], vxns[3];
	for (int i=0; i<3; i++)
	{
		Particle p(ovView.particles, objId*ovView.objSize+i);

		vs[i] = normalize(p.r - motion.r);
		assert(length(vs[i]) > 1e-3);

		ns[i] = cross(vs[i], make_float3(vs[i].y, vs[i].z, vs[i].x));
		vxns[i] = cross(vs[i], ns[i]);
	}

	float trqCoeffs[3];
	skewCoordinates(motion.torque, vxns, trqCoeffs);

	float3 parasiteForce = make_float3(0);
	for (int i=0; i<3; i++)
		parasiteForce += trqCoeffs[i] * ns[i];

	float frcCoeffs[3];
	skewCoordinates(motion.force-parasiteForce, vs, frcCoeffs);

	float3 resTorque = make_float3(0);
	float3 resForce  = make_float3(0);
	for (int i=0; i<3; i++)
	{
		float3 frc =  trqCoeffs[i] * ns[i] + frcCoeffs[i] * vs[i];
		resForce += frc;
		resTorque += cross(vs[i], frc);

		atomicAdd( ovView.forces + objId*ovView.objSize + i, frc );
	}

	if (length(resTorque - motion.torque) / max(length(resTorque) , 1.0f) > 1e-2 ||
		length(resForce  - motion.force)  / max(length(resForce),   1.0f) > 1e-2 )

		printf("%d:  frc [%f %f %f]  trq [%f %f %f]  ->  new frc [%f %f %f] new trq [%f %f %f], diff %f  %f\n",
				threadIdx.x, motion.force.x,  motion.force.y, motion.force.z,
				motion.torque.x, motion.torque.y, motion.torque.z,
				resForce.x, resForce.y, resForce.z,
				resTorque.x, resTorque.y, resTorque.z,
				length(resForce  - motion.force)  / max(length(motion.force),  1.0f),
				length(resTorque - motion.torque) / max(length(motion.torque), 1.0f));


	// Check that we conserve torque and force
//	assert( length(resTorque - motion.torque) / max(length(resTorque), 1.0f) < 1e-2 );
//	assert( length(resForce  - motion.force)  / max(length(resForce),  1.0f) < 1e-2 );
}

