#include <core/datatypes.h>
#include <core/scan.h>
#include <core/celllist.h>
#include <core/non_cached_rw.h>
#include <core/helper_math.h>

__global__ void blendStartSize(const uchar4* cellsSize, int4* cellsStart, const CellListInfo cinfo)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (4*gid >= cinfo.totcells) return;

	uchar4 sizes  = cellsSize [gid];

	cellsStart[gid] += make_int4(sizes.x << cinfo.blendingPower, sizes.y << cinfo.blendingPower,
								 sizes.z << cinfo.blendingPower, sizes.w << cinfo.blendingPower);
}

__global__ void computeCellSizes(const float4* xyzouvwo, const int n,
		const CellListInfo cinfo, uint* cellsSize)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	float4 coo = readNoCache(xyzouvwo + pid*2);//xyzouvwo[gid*2];

	int cid = cinfo.getCellId(coo);

	// No atomic for chars
	// Workaround: pad zeros around char in proper position and add as int
	// Care: BIG endian!

	// XXX: relying here only on redistribution
	if (coo.x > -900.0f)
	{
		const int addr = cid / 4;
		const int slot = cid % 4;
		const uint increment = 1 << (slot*8);

		atomicAdd(cellsSize + addr, increment);
	}
}

__global__ void rearrangeParticles(const CellListInfo cinfo, uint* cellsSize, const int* cellsStart,
		const int n, const float4* in_xyzouvwo, float4* out_xyzouvwo,
		bool rearrangeForces, const float4* in_forces, float4* out_forces)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int pid = gid / 2;
	const int sh  = gid % 2;  // sh = 0 copies coordinates, sh = 1 -- velocity
	if (pid >= n) return;

	int dstId;
	// instead of:
	// const float4 val = in_xyzouvwo[gid];
	//
	// this is to allow more cache for atomics
	// loads / stores here need no cache
	float4 val = readNoCache(in_xyzouvwo+gid);

	int cid;
	if (sh == 0)
	{
		cid = cinfo.getCellId(val);

		//  XXX: relying here only on redistribution
		if (val.x > -900.0f)
		{
			// See above
			const int addr = cid / 4;
			const int slot = cid % 4;
			const uint increment = 1 << (slot*8);

			const uint rawOffset = atomicSub(cellsSize + addr, increment);
			const int offset = ((rawOffset >> (slot*8)) & 255) - 1;

			int2 start_size = cinfo.decodeStartSize(cellsStart[cid]);
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

	if (dstId >= 0) writeNoCache(out_xyzouvwo + 2*dstId+sh, val);

	if (rearrangeForces && sh == 0)
	{
		float4 frc = readNoCache(in_forces + pid);
		writeNoCache(out_forces + dstId, frc);
	}
}


CellListInfo::CellListInfo(float rc, float3 domainSize) :
		rc(rc), h(make_float3(rc)), invh(make_float3(1.0/rc)), domainSize(domainSize)
{
	ncells = make_int3( ceilf(domainSize / rc - 1e-6) );
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
		CellListInfo(rc, domainSize), pv(pv)
{
	cellsStart.resize(totcells + 1);
	cellsSize .resize(totcells + 1);
}

CellList::CellList(ParticleVector* pv, int3 resolution, float3 domainSize) :
		CellListInfo(domainSize / make_float3(resolution), domainSize), pv(pv)
{
	cellsStart.resize(totcells + 1);
	cellsSize .resize(totcells + 1);
}

void CellList::build(cudaStream_t stream, bool rearrangeForces)
{
	if (pv->activeCL == this)
	{
		debug2("Cell-list for %s is already up-to-date, building skipped", pv->name.c_str());
		return;
	}

	if (pv->np >= (1<<blendingPower))
		die("Too many particles for the cell-list");

	if (pv->np / totcells >= (1<<(32-blendingPower)))
		die("Too many particles for the cell-list");

	if (pv->np == 0)
	{
		debug2("%s consists of no particles, cell-list building skipped", pv->name.c_str());
		return;
	}

	// Containers setup
	pv->pushStreamWOhalo(stream);

	// Compute cell sizes
	debug2("Computing cell sizes for %d %s particles", pv->np, pv->name.c_str());
	CUDA_Check( cudaMemsetAsync(cellsSize.devPtr(), 0, (totcells + 1)*sizeof(uint8_t), stream) );  // +1 to have correct cellsStart[totcells]

	auto cinfo = cellInfo();

	computeCellSizes<<< (pv->np+127)/128, 128, 0, stream >>> (
						(float4*)pv->coosvels.devPtr(), pv->np, cinfo, (uint*)cellsSize.devPtr());

	// Scan to get cell starts
	scan(cellsSize.devPtr(), totcells+1, cellsStart.devPtr(), stream);

	// Blend size and start together
	blendStartSize<<< ((totcells+3)/4 + 127) / 128, 128, 0, stream >>>((uchar4*)cellsSize.devPtr(), (int4*)cellsStart.devPtr(), cinfo);

	// Rearrange the data
	debug2("Rearranging %d %s particles", pv->np, pv->name.c_str());

	rearrangeParticles<<< (2*pv->np+127)/128, 128, 0, stream >>> (
			cinfo, (uint*)cellsSize.devPtr(), cellsStart.devPtr(),
			pv->np,
			(float4*)pv->coosvels.devPtr(), (float4*)pv->pingPongCoosvels.devPtr(),
			rearrangeForces,
			(float4*)pv->forces  .devPtr(), (float4*)pv->pingPongForces  .devPtr());


	// Now we need the new size of particles array.
	int newSize;
	CUDA_Check( cudaMemcpyAsync(&newSize, cellsStart.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
	CUDA_Check( cudaStreamSynchronize(stream) );
	debug2("Rearranging completed, new size of %s particle vector is %d", pv->name.c_str(), newSize);

	pv->resize(newSize, resizePreserve);
	CUDA_Check( cudaStreamSynchronize(stream) );
	containerSwap(pv->coosvels, pv->pingPongCoosvels);
	if (rearrangeForces)
		containerSwap(pv->forces, pv->pingPongForces);

	// Containers setup
	pv->popStreamWOhalo();

	pv->activeCL = this;
}
