#include "simple_stationary_wall.h"

#include <fstream>
#include <cmath>
#include <texture_types.h>
#include <cassert>

#include <core/logger.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/bounce_solver.h>

#include <core/utils/cuda_rng.h>

#include "stationary_walls/cylinder.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"

//===============================================================================================
// Removing kernels
//===============================================================================================

template<typename InsideWallChecker>
__global__ void collectRemaining(PVview view, float4* remaining, int* nRemaining, InsideWallChecker checker)
{
	const float tolerance = 1e-6f;

	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	Particle p(view.particles, pid);

	const float val = checker(p.r);

	if (val <= -tolerance)
	{
		const int ind = atomicAggInc(nRemaining);
		p.write2Float4(remaining, ind);
	}
}

template<typename InsideWallChecker>
__global__ void packRemainingObjects(OVviewWithExtraData view, char* output, int* nRemaining, InsideWallChecker checker)
{
	const float tolerance = 1e-6f;

	// One warp per object
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int objId = gid / warpSize;
	const int tid = gid % warpSize;

	if (objId >= view.nObjects) return;

	bool isRemaining = true;
	for (int i=tid; i < view.objSize; i++)
	{
		Particle p(view.particles, objId * view.objSize + i);
		if (checker(p.r) <= -tolerance)
		{
			isRemaining = false;
			break;
		}
	}

	if (!isRemaining) return;

	int dstObjId;
	if (tid == 0)
		dstObjId = atomicAggInc(nRemaining);
	dstObjId = __shfl(dstObjId, 0);


	Particle* dstAddr = (Particle*)(output + dstObjId * view.packedObjSize_byte);
	for (int i=tid; i < view.objSize; i+=warpSize)
		dstAddr[i] = Particle(view.particles, objId * view.objSize + i);

	view.packExtraData(objId, (char*)(dstAddr+view.objSize));
}

__global__ void unpackRemainingObjects(OVviewWithExtraData view, const char* input)
{
	// One warp per object
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int objId = gid / warpSize;
	const int tid = gid % warpSize;

	if (objId >= view.nObjects) return;

	Particle* srcAddr = (Particle*)(input + objId * view.packedObjSize_byte);
	for (int i=tid; i < view.objSize; i+=warpSize)
		((Particle*)view.particles)[objId * view.objSize + i] = srcAddr[i];

	view.unpackExtraData(objId, (char*)(srcAddr+view.objSize));
}
//===============================================================================================
// Boundary cells kernels
//===============================================================================================

template<typename InsideWallChecker>
__device__ __forceinline__ bool isCellOnBoundary(PVview view, float3 cornerCoo, float3 len, InsideWallChecker checker)
{
	// About maximum distance a particle can cover in one step
	const float tol = 0.25f;
	int pos = 0, neg = 0;

	for (int i=0; i<2; i++)
		for (int j=0; j<2; j++)
			for (int k=0; k<2; k++)
			{
				// Value in the cell corner
				const float3 shift = make_float3(i ? len.x : 0.0f, j ? len.y : 0.0f, k ? len.z : 0.0f);
				const float s = checker(cornerCoo + shift);

				if (s >  tol) pos++;
				if (s < -tol) neg++;
			}

	return (pos != 8 && neg != 8);
}

template<bool QUERY, typename InsideWallChecker>
__global__ void getBoundaryCells(PVview view, CellListInfo cinfo, int* nBoundaryCells, int* boundaryCells, InsideWallChecker checker)
{
	const int cid = blockIdx.x * blockDim.x + threadIdx.x;
	if (cid >= cinfo.totcells) return;

	int3 ind;
	cinfo.decode(cid, ind.x, ind.y, ind.z);
	float3 cornerCoo = -0.5f*cinfo.localDomainSize + make_float3(ind)*cinfo.h;

	if (isCellOnBoundary(view, cornerCoo, cinfo.h, checker))
	{
		int id = atomicAggInc(nBoundaryCells);
		if (!QUERY) boundaryCells[id] = cid;
	}
}

//===============================================================================================
// SDF bouncing kernel
//===============================================================================================

template<typename InsideWallChecker>
__global__ void bounceKernel(PVview view, const int* wallCells, const int nWallCells, CellListInfo cinfo, const float dt, InsideWallChecker checker)
{
	const int maxIters = 50;
	const float corrStep = (1.0f / (float)maxIters) * dt;

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nWallCells) return;
	const int cid = wallCells[tid];
	const int pstart = cinfo.cellStarts[cid];
	const int pend   = cinfo.cellStarts[cid+1];

	for (int pid = pstart; pid < pend; pid++)
	{
		Particle p(cinfo.particles, pid);
		if (checker(p.r) <= 0.0f) continue;

		float3 oldCoo = p.r - p.u*dt;

		for (int i=0; i<maxIters; i++)
		{
			if (checker(oldCoo) < 0.0f) break;
			oldCoo -= p.u*corrStep;
		}

		const float alpha = solveLinSearch([=] (float lambda) {
			return checker(oldCoo + (p.r-oldCoo)*lambda);
		});
		float3 candidate = (alpha >= 0.0f) ? oldCoo + alpha * (p.r - oldCoo) : oldCoo;

		if (checker(candidate) >= 0.0f)
		for (int i=0; i<maxIters; i++)
		{
			if (checker(candidate) < 0.0f) break;

			float3 rndShift;
				rndShift.x = Saru::mean0var1(p.r.x - floorf(p.r.x), p.i1+i, p.i1*p.i1);
				rndShift.y = Saru::mean0var1(rndShift.x,            p.i1+i, p.i1*p.i1);
				rndShift.z = Saru::mean0var1(rndShift.y,            p.i1+i, p.i1*p.i1);

				if (checker(candidate + 5.0f*rndShift*dt) < 0.0f)
				{
					candidate += 5.0f*rndShift*dt;
					break;
				}
		}

		p.r = candidate;
		p.u = -p.u;

		p.write2Float4(cinfo.particles, pid);
	}
}

//===============================================================================================
// Checking kernel
//===============================================================================================

template<typename InsideWallChecker>
__global__ void checkInside(PVview view, int* nInside, InsideWallChecker checker)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	Float3_int coo(view.particles[2*pid]);

	float v = checker(coo.v);

	if (v > 0) atomicAggInc(nInside);
}

//===============================================================================================
// Member functions
//===============================================================================================

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::setup(MPI_Comm& comm, DomainInfo domain)
{
	info("Setting up wall %s", name.c_str());

	CUDA_Check( cudaDeviceSynchronize() );
	MPI_Check( MPI_Comm_dup(comm, &wallComm) );

	insideWallChecker.setup(wallComm, domain);

	CUDA_Check( cudaDeviceSynchronize() );
}


template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::attach(ParticleVector* pv, CellList* cl)
{
	if (dynamic_cast<PrimaryCellList*>(cl) == nullptr)
		die("PVs should only be attached to walls with the primary cell-lists! "
				"Invalid combination: wall %s, pv %s", name.c_str(), pv->name.c_str());

	CUDA_Check( cudaDeviceSynchronize() );
	particleVectors.push_back(pv);
	cellLists.push_back(cl);
	nBounceCalls.push_back(0);

	PVview view(pv, pv->local());
	PinnedBuffer<int> nBoundaryCells(1);
	nBoundaryCells.clear(0);
	SAFE_KERNEL_LAUNCH(
			getBoundaryCells<true>,
			(cl->totcells + 127) / 128, 128, 0, 0,
			view, cl->cellInfo(), nBoundaryCells.devPtr(), nullptr, insideWallChecker.handler() );

	nBoundaryCells.downloadFromDevice(0);

	debug("Found %d boundary cells", nBoundaryCells[0]);
	auto bc = new DeviceBuffer<int>(nBoundaryCells[0]);

	nBoundaryCells.clear(0);
	SAFE_KERNEL_LAUNCH(
			getBoundaryCells<false>,
			(cl->totcells + 127) / 128, 128, 0, 0,
			view, cl->cellInfo(), nBoundaryCells.devPtr(), bc->devPtr(), insideWallChecker.handler() );

	boundaryCells.push_back(bc);
	CUDA_Check( cudaDeviceSynchronize() );
}



template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::removeInner(ParticleVector* pv)
{
	CUDA_Check( cudaDeviceSynchronize() );

	PinnedBuffer<int> nRemaining(1);
	nRemaining.clear(0);

	int oldSize = pv->local()->size();
	if (oldSize == 0) return;

	const int nthreads = 128;
	// Need a different path for objects
	ObjectVector* ov = dynamic_cast<ObjectVector*>(pv);
	if (ov == nullptr)
	{
		PVview view(pv, pv->local());
		PinnedBuffer<Particle> tmp(view.size);

		SAFE_KERNEL_LAUNCH(
				collectRemaining,
				getNblocks(view.size, nthreads), nthreads, 0, 0,
				view, (float4*)tmp.devPtr(), nRemaining.devPtr(), insideWallChecker.handler() );

		nRemaining.downloadFromDevice(0);
		std::swap(pv->local()->coosvels, tmp);
		int oldSize = pv->local()->size();
		pv->local()->resize(nRemaining[0], 0);
	}
	else
	{
		// Prepare temp storage for extra object data
		OVviewWithExtraData ovView(ov, ov->local(), 0);
		DeviceBuffer<char> tmp(ovView.nObjects * ovView.packedObjSize_byte);

		SAFE_KERNEL_LAUNCH(
				packRemainingObjects,
				getNblocks(ovView.nObjects*32, nthreads), nthreads, 0, 0,
				ovView,	tmp.devPtr(), nRemaining.devPtr(), insideWallChecker.handler() );

		// Copy temporary buffers back
		nRemaining.downloadFromDevice(0);
		ov->local()->resize_anew(nRemaining[0]);
		ovView = OVviewWithExtraData(ov, ov->local(), 0);
		SAFE_KERNEL_LAUNCH(
				unpackRemainingObjects,
				getNblocks(ovView.nObjects*32, nthreads), nthreads, 0, 0,
				ovView, tmp.devPtr() );
	}

	pv->haloValid = false;
	pv->redistValid = false;
	pv->cellListStamp++;

	info("Removed inner entities of %s, keeping %d out of %d particles",
			pv->name.c_str(), pv->local()->size(), oldSize);

	CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::bounce(float dt, cudaStream_t stream)
{
	for (int i=0; i<particleVectors.size(); i++)
	{
		auto pv = particleVectors[i];
		auto cl = cellLists[i];
		auto bc = boundaryCells[i];
		PVview view(pv, pv->local());

		debug2("Bouncing %d %s particles, %d boundary cells",
				pv->local()->size(), pv->name.c_str(), bc->size());

		const int nthreads = 64;
		SAFE_KERNEL_LAUNCH(
				bounceKernel,
				getNblocks(bc->size(), nthreads), nthreads, 0, stream,
				view, bc->devPtr(), bc->size(), cl->cellInfo(), dt, insideWallChecker.handler() );

		CUDA_Check( cudaPeekAtLastError() );
		nBounceCalls[i]++;
	}
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::check(cudaStream_t stream)
{
	const int nthreads = 128;
	for (int i=0; i<particleVectors.size(); i++)
	{
		auto pv = particleVectors[i];
		{
			nInside.clearDevice(stream);
			PVview view(pv, pv->local());
			SAFE_KERNEL_LAUNCH(
					checkInside,
					getNblocks(view.size, nthreads), nthreads, 0, stream,
					view, nInside.devPtr(), insideWallChecker.handler() );

			nInside.downloadFromDevice(stream);

			info("%d particles of %s are inside the wall %s", nInside[0], pv->name.c_str(), name.c_str());
		}
	}
}

template class SimpleStationaryWall<StationaryWall_Sphere>;
template class SimpleStationaryWall<StationaryWall_Cylinder>;
template class SimpleStationaryWall<StationaryWall_SDF>;




