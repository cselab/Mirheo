#include "datatypes.h"
#include "scan.h"
#include "celllist.h"
#include "non_cached_rw.h"

__global__ void computeCellsSizes(const float4* xyzouvwo, const int n, const float3 domainStart, const int3 ncells, const float invrc, uint* cellsSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= n) return;

	const int cid = getCellId(xyzouvwo[gid*2], domainStart, ncells, invrc);
	// No atomic for chars
	// Workaround: pad zeros around char in proper position and add as int
	// Care: BIG endian!

	const int addr = cid / 4;
	const int slot = cid % 4;
	const int increment = 1 << (slot*8);

	atomicAdd(cellsSize + addr, increment);
}

__global__ void blendStartSize(uchar4* cellsSize, int4* cellsStart, const int totcells)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (4*gid >= totcells) return;

	uchar4 sizes  = cellsSize [gid];

	cellsStart[gid] += make_int4(sizes.x << 26, sizes.y << 26, sizes.z << 26, sizes.w << 26);
}

__global__ void rearrangeParticles(const float4* in_xyzouvwo, const int n, const float3 domainStart, const int3 ncells,
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
	const float4 val = readNoCache(in_xyzouvwo+gid);

	if (sh == 0)
	{
		const int cid = getCellId(val, domainStart, ncells, invrc);

		// See above
		const int addr = cid / 4;
		const int slot = cid % 4;
		const int increment = 1 << (slot*8);

		const int rawOffset = atomicAdd(cellsSize + addr, -increment);
		const int offset = ((rawOffset >> (slot*8)) & 255) - 1;

		dstId = ( cellsStart[cid] & ((1<<26) - 1) ) + offset;  // mask blended Start
	}
	int other = __shfl_up(dstId, 1);
	dstId = (sh == 0) ? 2*dstId : 2*other + 1;

	writeNoCache(out_xyzouvwo+dstId, val);
}


void buildCellList(float4* in_xyzouvwo, const int n, const float3 domainStart, const int3 ncells, const int totcells, const float invrc,
				   float4* out_xyzouvwo, uint8_t* cellsSize, int* cellsStart, cudaStream_t stream)
{
	// Compute cell sizes
	CUDA_Check( cudaMemsetAsync(cellsSize, 0, (totcells + 1)*sizeof(uint8_t), stream) );  // +1 to have correct cellsStart[totcells]
	computeCellsSizes<<< (n+127)/128, 128, 0, stream >>>(in_xyzouvwo, n, domainStart, ncells, invrc, (uint*)cellsSize);

	// Scan to get cell starts
	scan(cellsSize, totcells+1, cellsStart, stream);

	// Blend size and start together
	blendStartSize<<< (totcells/4 + 127) / 128, 128, 0, stream >>>((uchar4*)cellsSize, (int4*)cellsStart, totcells);

	// Rearrange the data
	rearrangeParticles<<< (2*n+127)/128, 128, 0, stream >>>(in_xyzouvwo, n, domainStart, ncells, invrc, (uint*)cellsSize, cellsStart, out_xyzouvwo);
}
