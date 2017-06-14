#include <core/particle_vector.h>
#include <core/object_vector.h>
#include <core/scan.h>
#include <core/celllist.h>
#include <core/cuda_common.h>
#include <core/helper_math.h>

__global__ void blendStartSize(const uchar4* cellsSize, uint4* cellsStartSize, const CellListInfo cinfo)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (4*gid >= cinfo.totcells) return;

	uchar4 sizes  = cellsSize [gid];

	cellsStartSize[gid] += make_uint4(sizes.x << cinfo.blendingPower, sizes.y << cinfo.blendingPower,
								  sizes.z << cinfo.blendingPower, sizes.w << cinfo.blendingPower);
}

__global__ void computeCellSizes(const float4* coosvels, const int n,
		const CellListInfo cinfo, uint* cellsSize)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	float4 coo = readNoCache(coosvels + pid*2);//coosvels[gid*2];

	int cid = cinfo.getCellId(coo);

	// No atomic for chars
	// Workaround: pad zeros around char in proper position and add as int
	// Care: BIG endian!

	// XXX: relying here only on redistribution
	if (coo.x > -900.0f)
	{
		const uint addr = cid / 4;
		const uint slot = cid % 4;
		const uint increment = 1 << (slot*8);

		uint tmp = atomicAdd(cellsSize + addr, increment);

		tmp = tmp >> (slot*8);
		tmp = tmp & 0xff;

		if (tmp > 200) printf("%d \n", tmp);
	}
}

__global__ void reorderParticles(const CellListInfo cinfo, uint* cellsSize, const uint* cellsStartSize,
		const int n, const float4* in_coosvels, float4* out_coosvels, int* order)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 copies coordinates, sh = 1 -- velocity
	if (pid >= n) return;

	int dstId;
	// instead of:
	// const float4 val = in_coosvels[gid];
	//
	// this is to allow more cache for atomics
	// loads / stores here need no cache
	float4 val = readNoCache(in_coosvels+gid);

	int cid;
	if (sh == 0)
	{
		cid = cinfo.getCellId(val);

		//  XXX: relying here only on redistribution
		if (val.x > -900.0f)
		{
			// See above
			const uint addr = cid / 4;
			const uint slot = cid % 4;
			const uint increment = 1 << (slot*8);

			const uint rawOffset = atomicSub(cellsSize + addr, increment);
			const int offset = ((rawOffset >> (slot*8)) & 255) - 1;

			int2 start_size = cinfo.decodeStartSize(cellsStartSize[cid]);
			dstId = start_size.x + offset;  // mask blended Start
		}
		else
		{
			dstId = -1;
		}
	}

	int otherDst = __shfl_up(dstId, 1);
	if (sh == 1)
		dstId = otherDst;

	if (dstId >= 0)
	{
		writeNoCache(out_coosvels + 2*dstId+sh, val);
		if (sh == 0) order[pid] = dstId;
	}
}

__global__ void addForcesKernel(const int n, const float4* in_forces, float4* out_forces, int* order)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	out_forces[pid] += in_forces[order[pid]];
}

CellListInfo::CellListInfo(float rc, float3 domainSize) :
		rc(rc), h(make_float3(rc)), domainSize(domainSize)
{
	ncells = make_int3( floorf(domainSize / rc + 1e-6) );
	float3 h = make_float3(domainSize) / make_float3(ncells);
	invh = 1.0f / h;
	this->rc = std::min( {h.x, h.y, h.z} );

	totcells = ncells.x * ncells.y * ncells.z;
}

CellListInfo::CellListInfo(float3 h, float3 domainSize) :
		h(h), invh(1.0f/h), domainSize(domainSize)
{
	rc = std::min( {h.x, h.y, h.z} );
	ncells = make_int3( ceilf(domainSize / h - 1e-6f) );
	totcells = ncells.x * ncells.y * ncells.z;
}


CellList::CellList(ParticleVector* pv, float rc, float3 domainSize) :
		CellListInfo(rc, domainSize), pv(pv), stream(0)
{
	cellsStartSize.resize(totcells + 1);
	cellsSize .resize(totcells + 1);

	debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}

CellList::CellList(ParticleVector* pv, int3 resolution, float3 domainSize) :
		CellListInfo(domainSize / make_float3(resolution), domainSize), pv(pv), stream(0)
{
	cellsStartSize.resize(totcells + 1);
	cellsSize .resize(totcells + 1);

	debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}

void CellList::build()
{
	if (pv->local()->changedStamp == changedStamp)
	{
		debug2("Cell-list for %s is already up-to-date, building skipped", pv->name.c_str());
		return;
	}

	if (pv->local()->size() >= (1<<blendingPower))
		die("Too many particles for the cell-list");

	if (pv->local()->size() / totcells >= (1<<(32-blendingPower)))
		die("Too many particles for the cell-list");

	if (pv->local()->size() == 0)
	{
		debug2("%s consists of no particles, cell-list building skipped", pv->name.c_str());
		return;
	}

	// Containers setup
	pv->local()->pushStream(stream);

	// Compute cell sizes
	debug2("Computing cell sizes for %d %s particles", pv->local()->size(), pv->name.c_str());
	CUDA_Check( cudaMemsetAsync(cellsSize.devPtr(), 0, (totcells + 1)*sizeof(uint8_t), stream) );  // +1 to have correct cellsStartSize[totcells]

	computeCellSizes<<< (pv->local()->size()+127)/128, 128, 0, stream >>> (
						(float4*)pv->local()->coosvels.devPtr(), pv->local()->size(), cellInfo(), (uint*)cellsSize.devPtr());

	// Scan to get cell starts
	scan(cellsSize.devPtr(), totcells+1, (int*)cellsStartSize.devPtr(), stream);

	// Blend size and start together
	blendStartSize<<< ((totcells+3)/4 + 127) / 128, 128, 0, stream >>>((uchar4*)cellsSize.devPtr(), (uint4*)cellsStartSize.devPtr(), cellInfo());

	// Reorder the data
	debug2("Reordering %d %s particles", pv->local()->size(), pv->name.c_str());
	order.resize(pv->local()->size());
	_coosvels.resize(pv->local()->size());
	_forces.resize(pv->local()->size());

	reorderParticles<<< (2*pv->local()->size()+127)/128, 128, 0, stream >>> (
			cellInfo(), (uint*)cellsSize.devPtr(), cellsStartSize.devPtr(),
			pv->local()->size(), (float4*)pv->local()->coosvels.devPtr(), (float4*)_coosvels.devPtr(), order.devPtr());

	if (primary)
	{
		// Now we need the new size of particles array.
		int newSize;
		CUDA_Check( cudaMemcpyAsync(&newSize, cellsStartSize.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
		CUDA_Check( cudaStreamSynchronize(stream) );
		newSize = newSize & ((1<<blendingPower) - 1);
		debug2("Reordering completed, new size of %s particle vector is %d", pv->name.c_str(), newSize);

		pv->local()->resize(newSize, resizePreserve);
		_coosvels.resize(newSize, resizePreserve);
		CUDA_Check( cudaStreamSynchronize(stream) );

		containerSwap(pv->local()->coosvels, _coosvels);
		coosvels = &pv->local()->coosvels;
		forces   = &pv->local()->forces;
	}
	else
	{
		coosvels = &_coosvels;
		forces   = &_forces;
	}

	// Containers setup
	pv->local()->popStream();

	changedStamp = pv->local()->changedStamp;
}

void CellList::addForces()
{
	if (forces != &pv->local()->forces)
		addForcesKernel<<< (pv->local()->size()+127)/128, 128, 0, stream >>> (pv->local()->size(), (float4*)forces->devPtr(), (float4*)pv->local()->forces.devPtr(), order.devPtr());
}
