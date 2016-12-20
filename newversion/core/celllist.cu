#include "datatypes.h"
#include "scan.h"
#include "celllist.h"
#include "non_cached_rw.h"

__global__ void blendStartSize(uchar4* cellsSize, int4* cellsStart, const int totcells)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (4*gid >= totcells) return;

	uchar4 sizes  = cellsSize [gid];

	cellsStart[gid] += make_int4(sizes.x << 26, sizes.y << 26, sizes.z << 26, sizes.w << 26);
}

__global__ void computeCellSizes(const float4* xyzouvwo, const int n, const int nMovable,
		const float3 domainStart, const int3 ncells, const float invrc, uint* cellsSize)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	float4 coo = readNoCache(xyzouvwo + pid*2);//xyzouvwo[gid*2];

	int cid;
	if (pid < nMovable)
		cid = getCellId<false>(coo, domainStart, ncells, invrc);
	else
		cid = getCellId(coo, domainStart, ncells, invrc);

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

__global__ void rearrangeParticles(const float4* in_xyzouvwo, const int n, const int nMovable, const float3 domainStart, const int3 ncells,
		const float invrc, uint* cellsSize, int* cellsStart, float4* out_xyzouvwo)
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
			cid = getCellId<false>(val, domainStart, ncells, invrc);
		else
			cid = getCellId(val, domainStart, ncells, invrc);

		if (cid >= 0)
		{
			// See above
			const int addr = cid / 4;
			const int slot = cid % 4;
			const int increment = 1 << (slot*8);

			const int rawOffset = atomicAdd(cellsSize + addr, -increment);
			const int offset = ((rawOffset >> (slot*8)) & 255) - 1;

			int2 start_size = decodeStartSize(cellsStart[cid]);
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


void buildCellList(ParticleVector& pv, cudaStream_t stream)
{
	// Compute cell sizes
	debug("Computing cell sizes for %d particles with %d newcomers", pv.np, pv.received);
	CUDA_Check( cudaMemsetAsync(pv.cellsSize.devdata, 0, (pv.totcells + 1)*sizeof(uint8_t), stream) );  // +1 to have correct cellsStart[totcells]

	computeCellSizes<<< (pv.np+127)/128, 128, 0, stream >>> (
						(float4*)pv.coosvels.devdata, pv.np, pv.np - pv.received, pv.domainStart, pv.ncells, 1.0f, (uint*)pv.cellsSize.devdata);

	// Scan to get cell starts
	scan(pv.cellsSize.devdata, pv.totcells+1, pv.cellsStart.devdata, stream);

	// Blend size and start together
	blendStartSize<<< ((pv.totcells+3)/4 + 127) / 128, 128, 0, stream >>>((uchar4*)pv.cellsSize.devdata, (int4*)pv.cellsStart.devdata, pv.totcells);

	// Rearrange the data
	debug("Rearranging %d particles", pv.np);

	rearrangeParticles<<< (2*pv.np+127)/128, 128, 0, stream >>> (
						(float4*)pv.coosvels.devdata, pv.np, pv.np - pv.received, pv.domainStart, pv.ncells, 1.0f,
						(uint*)pv.cellsSize.devdata, pv.cellsStart.devdata, (float4*)pv.pingPongBuf.devdata);

	debug("Rearranging completed");

	// Now we need the new size of particles array.
	int newSize;
	CUDA_Check( cudaMemcpyAsync(&newSize, pv.cellsStart.devdata + pv.totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
	CUDA_Check( cudaStreamSynchronize(stream) );
	pv.resize(newSize, resizePreserve, stream);

	swap(pv.coosvels, pv.pingPongBuf, stream);
}
