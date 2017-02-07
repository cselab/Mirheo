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

	cellsStart[gid] += make_int4(sizes.x << 26, sizes.y << 26, sizes.z << 26, sizes.w << 26);
}

__global__ void computeCellSizes(const float4* xyzouvwo, const int n, const int nMovable,
		const CellListInfo cinfo, uint* cellsSize)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	float4 coo = readNoCache(xyzouvwo + pid*2);//xyzouvwo[gid*2];

	int cid;
	if (pid < nMovable)
		cid = cinfo.getCellId<false>(coo);
	else
		cid = cinfo.getCellId(coo);

	// No atomic for chars
	// Workaround: pad zeros around char in proper position and add as int
	// Care: BIG endian!
	if (cid >= 0)
	{
		const int addr = cid / 4;
		const int slot = cid % 4;
		const int increment = 1 << (slot*8);

		atomicAdd(cellsSize + addr, increment);
	}
}

__global__ void rearrangeParticles(const float4* in_xyzouvwo, const int n, const int nMovable,
		const CellListInfo cinfo, uint* cellsSize, const int* cellsStart, float4* out_xyzouvwo)
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
		if (pid < nMovable)
			cid = cinfo.getCellId<false>(val);
		else
			cid = cinfo.getCellId(val);

		if (cid >= 0)
		{
			// See above
			const int addr = cid / 4;
			const int slot = cid % 4;
			const int increment = 1 << (slot*8);

			const int rawOffset = atomicAdd(cellsSize + addr, -increment);
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
}


CellListInfo::CellListInfo(float rc, float3 domainStart, float3 length) :
		rc(rc), invrc(1.0f/rc), domainStart(domainStart), length(length)
{
	ncells = make_int3( ceilf(length / rc - 1e-6) );
	totcells = ncells.x * ncells.y * ncells.z;
}


CellList::CellList(ParticleVector* pv, float rc, float3 domainStart, float3 length) :
		CellListInfo(rc, domainStart, length), pv(pv)
{
	cellsStart.resize(totcells + 1);
	cellsSize.resize(totcells + 1);
}

CellList::CellList(ParticleVector* pv, int3 resolution, float3 domainStart, float3 length) :
		CellListInfo(1.0f, domainStart, length), pv(pv)
{
	ncells = resolution;
	totcells = ncells.x * ncells.y * ncells.z;
	rc = std::min({length.x / ncells.x, length.y / ncells.y, length.z / ncells.z});
	cellsStart.resize(totcells + 1);
	cellsSize.resize(totcells + 1);
}

void CellList::build(cudaStream_t stream)
{
	// Containers setup
	pv->pushStreamWOhalo(stream);

	// Compute cell sizes
	debug("Computing cell sizes for %d particles with %d newcomers", pv->np, pv->received);
	CUDA_Check( cudaMemsetAsync(cellsSize.devPtr(), 0, (totcells + 1)*sizeof(uint8_t), stream) );  // +1 to have correct cellsStart[totcells]

	auto cinfo = cellInfo();

	computeCellSizes<<< (pv->np+127)/128, 128, 0, stream >>> (
						(float4*)pv->coosvels.devPtr(), pv->np, pv->np - pv->received, cinfo, (uint*)cellsSize.devPtr());

	// Scan to get cell starts
	scan(cellsSize.devPtr(), totcells+1, cellsStart.devPtr(), stream);

	// Blend size and start together
	blendStartSize<<< ((totcells+3)/4 + 127) / 128, 128, 0, stream >>>((uchar4*)cellsSize.devPtr(), (int4*)cellsStart.devPtr(), cinfo);

	// Rearrange the data
	debug("Rearranging %d particles", pv->np);

	rearrangeParticles<<< (2*pv->np+127)/128, 128, 0, stream >>> (
						(float4*)pv->coosvels.devPtr(), pv->np, pv->np - pv->received, cinfo,
						(uint*)cellsSize.devPtr(), cellsStart.devPtr(), (float4*)pv->pingPongBuf.devPtr());


	// Now we need the new size of particles array.
	int newSize;
	CUDA_Check( cudaMemcpyAsync(&newSize, cellsStart.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
	CUDA_Check( cudaStreamSynchronize(stream) );
	debug("Rearranging completed, new size of particle vector is %d", newSize);

	pv->resize(newSize, resizePreserve);
	CUDA_Check( cudaStreamSynchronize(stream) );
	containerSwap(pv->coosvels, pv->pingPongBuf);

	// Containers setup
	pv->popStreamWOhalo();
}
