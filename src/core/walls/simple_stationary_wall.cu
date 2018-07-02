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
#include <core/pvs/extra_data/packers.h>
#include <core/bounce_solver.h>

#include <core/utils/cuda_rng.h>

#include "stationary_walls/cylinder.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/box.h"

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
__global__ void packRemainingObjects(OVview view, ObjectPacker packer, char* output, int* nRemaining, InsideWallChecker checker)
{
	const float tolerance = 1e-6f;

	// One warp per object
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int objId = gid / warpSize;
	const int tid = gid % warpSize;

	if (objId >= view.nObjects) return;

	bool isRemaining = true;
	for (int i=tid; i < view.objSize; i+=warpSize)
	{
		Particle p(view.particles, objId * view.objSize + i);
		if (checker(p.r) > -tolerance)
		{
			isRemaining = false;
			break;
		}
	}

	isRemaining = __all(isRemaining);
	if (!isRemaining) return;

	int dstObjId;
	if (tid == 0)
		dstObjId = atomicAdd(nRemaining, 1);
	dstObjId = __shfl(dstObjId, 0);

	char* dstAddr = output + dstObjId * packer.totalPackedSize_byte;
	for (int pid = tid; pid < view.objSize; pid += warpSize)
	{
		const int srcPid = objId * view.objSize + pid;
		packer.part.pack(srcPid, dstAddr + pid*packer.part.packedSize_byte);
	}

	dstAddr += view.objSize * packer.part.packedSize_byte;
	if (tid == 0) packer.obj.pack(objId, dstAddr);
}

__global__ static void unpackRemainingObjects(const char* from, OVview view, ObjectPacker packer)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;

	const char* srcAddr = from + packer.totalPackedSize_byte * objId;

	for (int pid = tid; pid < view.objSize; pid += blockDim.x)
	{
		const int dstId = objId*view.objSize + pid;
		packer.part.unpack(srcAddr + pid*packer.part.packedSize_byte, dstId);
	}

	srcAddr += view.objSize * packer.part.packedSize_byte;
	if (tid == 0) packer.obj.unpack(srcAddr, objId);
}
//===============================================================================================
// Boundary cells kernels
//===============================================================================================

template<typename InsideWallChecker>
__device__ inline bool isCellOnBoundary(PVview view, float3 cornerCoo, float3 len, InsideWallChecker checker)
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
__device__ inline float3 rescue(float3 candidate, float dt, float tol, int id, const InsideWallChecker& checker)
{
	const int maxIters = 20;
	const float factor = 5.0f*dt;

	for (int i=0; i<maxIters; i++)
	{
		float v = checker(candidate);
		if (v < -tol) break;

		float3 rndShift;
		rndShift.x = Saru::mean0var1(candidate.x - floorf(candidate.x), id+i, id*id);
		rndShift.y = Saru::mean0var1(rndShift.x,                        id+i, id*id);
		rndShift.z = Saru::mean0var1(rndShift.y,                        id+i, id*id);

		if (checker(candidate + factor*rndShift) < v)
			candidate += factor*rndShift;
	}

	return candidate;
}

template<typename InsideWallChecker>
__global__ void bounceKernel(
		PVviewWithOldParticles view, CellListInfo cinfo,
		const int* wallCells, const int nWallCells, const float dt, const InsideWallChecker checker)
{
	const float tol = 2e-6f;

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nWallCells) return;
	const int cid = wallCells[tid];
	const int pstart = cinfo.cellStarts[cid];
	const int pend   = cinfo.cellStarts[cid+1];

	for (int pid = pstart; pid < pend; pid++)
	{
		Particle p(view.particles, pid);
		if (checker(p.r) <= -tol) continue;

		Particle pOld(view.old_particles, pid);
		float3 dr = p.r - pOld.r;

		const float alpha = solveLinSearch([=] (float lambda) {
			return checker(pOld.r + dr*lambda) + tol;
		});
		float3 candidate = (alpha >= 0.0f) ? pOld.r + alpha * dr : pOld.r;
		candidate = rescue(candidate, dt, tol, p.i1, checker);

		p.r = candidate;
		p.u = -p.u;

		p.write2Float4(cinfo.particles, pid);
	}
}

//===============================================================================================
// Checking kernel
//===============================================================================================

template<typename InsideWallChecker>
__global__ void checkInside(PVview view, int* nInside, const InsideWallChecker checker)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	Float3_int coo(view.particles[2*pid]);

	float v = checker(coo.v);

	if (v > 0) atomicAggInc(nInside);
}

//===============================================================================================
// Kernels computing sdf and sdf gradient per particle
//===============================================================================================

template<typename InsideWallChecker>
__global__ void computeSdfPerParticle(PVview view, float* sdfs, float3* gradients, InsideWallChecker checker)
{
	const float h = 0.25f;

	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	Particle p;
	p.readCoordinate(view.particles, pid);

	float sdf = checker(p.r);
	sdfs[pid] = sdf;

	if (gradients != nullptr)
	{
		float sdf_mx = checker(p.r + make_float3(-h,  0,  0));
		float sdf_px = checker(p.r + make_float3( h,  0,  0));
		float sdf_my = checker(p.r + make_float3( 0, -h,  0));
		float sdf_py = checker(p.r + make_float3( 0,  h,  0));
		float sdf_mz = checker(p.r + make_float3( 0,  0, -h));
		float sdf_pz = checker(p.r + make_float3( 0,  0,  h));

		float3 grad = make_float3( sdf_px - sdf_mx, sdf_py - sdf_my, sdf_pz - sdf_mz ) * (1.0f / (2.0f*h));

		gradients[pid] = normalize(grad);
	}
}

//===============================================================================================
// Member functions
//===============================================================================================

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::setup(MPI_Comm& comm, DomainInfo domain, ParticleVector* jointPV)
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
		pv->local()->resize(nRemaining[0], 0);
	}
	else
	{
		// Prepare temp storage for extra object data
		OVview ovView(ov, ov->local());
		ObjectPacker packer(ov, ov->local(), 0);

		DeviceBuffer<char> tmp(ovView.nObjects * packer.totalPackedSize_byte);

		SAFE_KERNEL_LAUNCH(
				packRemainingObjects,
				getNblocks(ovView.nObjects*32, nthreads), nthreads, 0, 0,
				ovView,	packer, tmp.devPtr(), nRemaining.devPtr(), insideWallChecker.handler() );

		// Copy temporary buffers back
		nRemaining.downloadFromDevice(0);
		ov->local()->resize_anew(nRemaining[0] * ov->objSize);
		ovView = OVview(ov, ov->local());
		packer = ObjectPacker(ov, ov->local(), 0);

		SAFE_KERNEL_LAUNCH(
				unpackRemainingObjects,
				ovView.nObjects, nthreads, 0, 0,
				tmp.devPtr(), ovView, packer  );
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
		PVviewWithOldParticles view(pv, pv->local());

		debug2("Bouncing %d %s particles, %d boundary cells",
				pv->local()->size(), pv->name.c_str(), bc->size());

		const int nthreads = 64;
		SAFE_KERNEL_LAUNCH(
				bounceKernel,
				getNblocks(bc->size(), nthreads), nthreads, 0, stream,
				view, cl->cellInfo(), bc->devPtr(), bc->size(), dt, insideWallChecker.handler() );

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

			say("%d particles of %s are inside the wall %s", nInside[0], pv->name.c_str(), name.c_str());
		}
	}
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfPerParticle(LocalParticleVector* lpv, GPUcontainer* sdfs, GPUcontainer* gradients, cudaStream_t stream)
{
	const int nthreads = 128;

	auto pv = lpv->pv;
	if (sdfs->size() * sdfs->datatype_size() < sizeof(float) * lpv->size())
		die("Provided too small container for SDF values of PV '%s'", pv->name.c_str());

	if (gradients != nullptr)
		if (gradients->size() * gradients->datatype_size() < sizeof(float3) * lpv->size())
			die("Provided too small container for SDF gradients of PV '%s'", pv->name.c_str());

	PVview view(pv, lpv);
	SAFE_KERNEL_LAUNCH(
			computeSdfPerParticle,
			getNblocks(view.size, nthreads), nthreads, 0, stream,
			view, (float*)sdfs->genericDevPtr(),
			(gradients != nullptr) ? (float3*)gradients->genericDevPtr() : nullptr, insideWallChecker.handler() );
}



template class SimpleStationaryWall<StationaryWall_Sphere>;
template class SimpleStationaryWall<StationaryWall_Cylinder>;
template class SimpleStationaryWall<StationaryWall_SDF>;
template class SimpleStationaryWall<StationaryWall_Plane>;
template class SimpleStationaryWall<StationaryWall_Box>;




