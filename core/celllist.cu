#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/logger.h>

#include <core/cub/device/device_scan.cuh>


__global__ void computeCellSizes(PVview view, CellListInfo cinfo)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	float4 coo = readNoCache(view.particles + pid*2);//coosvels[gid*2];
	int cid = cinfo.getCellId(coo);

	// XXX: relying here only on redistribution
	if (coo.x > -900.0f)
		atomicAdd(cinfo.cellSizes + cid, 1);
}

// TODO: use old_particles as buffer
__global__ void reorderParticles(PVview view, CellListInfo cinfo, float4* outParticles)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 copies coordinates, sh = 1 -- velocity
	if (pid >= view.size) return;

	int dstId;

	// instead of:
	// const float4 val = in_coosvels[gid];
	//
	// this is to allow more cache for atomics
	// loads / stores here need no cache
	float4 val = readNoCache(view.particles+gid);

	int cid;
	if (sh == 0)
	{
		cid = cinfo.getCellId(val);

		//  XXX: relying here only on redistribution
		if (val.x > -900.0f)
			dstId = cinfo.cellStarts[cid] + atomicAdd(cinfo.cellSizes + cid, 1);
		else
			dstId = -1;
	}

	int otherDst = __shfl_up(dstId, 1);
	if (sh == 1)
		dstId = otherDst;

	if (dstId >= 0)
	{
		writeNoCache(outParticles + 2*dstId+sh, val);
		if (sh == 0) cinfo.order[pid] = dstId;
	}
}

__global__ void addForcesKernel(PVview view, CellListInfo cinfo)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= view.size) return;

	view.forces[pid] += cinfo.forces[cinfo.order[pid]];
}

//=================================================================================
// Info
//=================================================================================

CellListInfo::CellListInfo(float rc, float3 localDomainSize) :
		rc(rc), h(make_float3(rc)), localDomainSize(localDomainSize)
{
	ncells = make_int3( floorf(localDomainSize / rc + 1e-6) );
	float3 h = make_float3(localDomainSize) / make_float3(ncells);
	invh = 1.0f / h;
	this->rc = std::min( {h.x, h.y, h.z} );

	totcells = ncells.x * ncells.y * ncells.z;
}

CellListInfo::CellListInfo(float3 h, float3 localDomainSize) :
		h(h), invh(1.0f/h), localDomainSize(localDomainSize)
{
	rc = std::min( {h.x, h.y, h.z} );
	ncells = make_int3( ceilf(localDomainSize / h - 1e-6f) );
	totcells = ncells.x * ncells.y * ncells.z;
}

//=================================================================================
// Basic cell-lists
//=================================================================================

CellList::CellList(ParticleVector* pv, float rc, float3 localDomainSize) :
		CellListInfo(rc, localDomainSize), pv(pv),
		particles(&particlesContainer),
		forces(&forcesContainer)
{
	cellSizes. resize_anew(totcells + 1);
	cellStarts.resize_anew(totcells + 1);

	debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}

CellList::CellList(ParticleVector* pv, int3 resolution, float3 localDomainSize) :
		CellListInfo(localDomainSize / make_float3(resolution), localDomainSize), pv(pv),
		particles(&particlesContainer),
		forces(&forcesContainer)
{
	cellSizes. resize_anew(totcells + 1);
	cellStarts.resize_anew(totcells + 1);

	debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}

void CellList::_build(cudaStream_t stream)
{
	// Compute cell sizes
	debug2("Computing cell sizes for %d %s particles", pv->local()->size(), pv->name.c_str());
	cellSizes.clear(stream);

	PVview view(pv, pv->local());

	int nthreads = 128;
	SAFE_KERNEL_LAUNCH(
			computeCellSizes,
			getNblocks(view.size, nthreads), nthreads, 0, stream,
			view, cellInfo() );

	// Scan to get cell starts
	size_t bufSize;
	cub::DeviceScan::ExclusiveSum(nullptr, bufSize, cellSizes.devPtr(), cellStarts.devPtr(), totcells+1, stream);
	// Allocate temporary storage
	scanBuffer.resize_anew(bufSize);
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(scanBuffer.devPtr(), bufSize, cellSizes.devPtr(), cellStarts.devPtr(), totcells+1, stream);

	// Reorder the data
	debug2("Reordering %d %s particles", pv->local()->size(), pv->name.c_str());
	order.resize_anew(view.size);
	particlesContainer.resize_anew(view.size);
	cellSizes.clear(stream);

	SAFE_KERNEL_LAUNCH(
			reorderParticles,
			getNblocks(2*view.size, nthreads), nthreads, 0, stream,
			view, cellInfo(), (float4*)particlesContainer.devPtr() );

	changedStamp = pv->cellListStamp;
}

void CellList::build(cudaStream_t stream)
{
	if (changedStamp == pv->cellListStamp)
	{
		debug2("Cell-list for %s is already up-to-date, building skipped", pv->name.c_str());
		return;
	}

	if (pv->local()->size() == 0)
	{
		debug2("%s consists of no particles, cell-list building skipped", pv->name.c_str());
		return;
	}

	_build(stream);

	forcesContainer.resize_anew(pv->local()->size());
}

void CellList::addForces(cudaStream_t stream)
{
	PVview view(pv, pv->local());
	int nthreads = 128;

	SAFE_KERNEL_LAUNCH(
			addForcesKernel,
			getNblocks(view.size, nthreads), nthreads, 0, stream,
			view, cellInfo() );
}

//=================================================================================
// Primary cell-lists
//=================================================================================

PrimaryCellList::PrimaryCellList(ParticleVector* pv, float rc, float3 localDomainSize) :
		CellList(pv, rc, localDomainSize)
{
	particles = &pv->local()->coosvels;
	forces    = &pv->local()->forces;

	if (dynamic_cast<ObjectVector*>(pv) != nullptr)
		error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

PrimaryCellList::PrimaryCellList(ParticleVector* pv, int3 resolution, float3 localDomainSize) :
		CellList(pv, resolution, localDomainSize)
{
	particles = &pv->local()->coosvels;
	forces    = &pv->local()->forces;

	if (dynamic_cast<ObjectVector*>(pv) != nullptr)
		error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

void PrimaryCellList::build(cudaStream_t stream)
{
	//warn("Reordering extra data is not yet implemented in cell-lists");

	if (changedStamp == pv->cellListStamp)
	{
		debug2("Cell-list for %s is already up-to-date, building skipped", pv->name.c_str());
		return;
	}

	if (pv->local()->size() == 0)
	{
		debug2("%s consists of no particles, cell-list building skipped", pv->name.c_str());
		return;
	}

	_build(stream);

	// Now we need the new size of particles array.
	int newSize;
	CUDA_Check( cudaMemcpyAsync(&newSize, cellStarts.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
	CUDA_Check( cudaStreamSynchronize(stream) );

	debug2("Reordering completed, new size of %s particle vector is %d", pv->name.c_str(), newSize);

	particlesContainer.resize(newSize, stream);
	pv->local()->resize_anew(newSize);
	std::swap(pv->local()->coosvels, particlesContainer);
}













