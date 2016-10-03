#include "datatypes.h"
#include "scan.h"
#include "celllist.h"

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
	// this is to avoid allow more cache for atomics
	// loads / stores here need no cache
	float4 val;
	const float4* src = in_xyzouvwo+gid;
	asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(src));

	if (sh == 0)
	{
		const int cid = getCellId(val, domainStart, ncells, invrc);

		// See above
		const int addr = cid / 4;
		const int slot = cid % 4;
		const int increment = 1 << (slot*8);

		const int rawOffset = atomicAdd(cellsSize + addr, -increment);
		const int offset = ((rawOffset >> (slot*8)) & 255) - 1;

		dstId = cellsStart[cid] + offset;
	}
	int other = __shfl_up(dstId, 1);
	dstId = (sh == 0) ? 2*dstId : 2*other + 1;

	//out_xyzouvwo[dstId] = val;
	float4* dest = out_xyzouvwo + dstId;
	asm("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(dest), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
}

void buildCellListWithPrecomputedSizes(float4* in_xyzouvwo, const int n, const float3 domainStart, const int3 ncells, const float invrc,
				   float4* out_xyzouvwo, uint8_t* cellsSize, int* cellsStart, cudaStream_t stream)
{
	// cellsSize assumed to be properly filled
	const int totcells = ncells.x*ncells.y*ncells.z;

	// Scan to get cell starts
	scan(cellsSize, totcells+1, cellsStart, stream);

	// Rearrange the data
	rearrangeParticles<<< (2*n+127)/128, 128, 0, stream >>>(in_xyzouvwo, n, domainStart, ncells, invrc, (uint*)cellsSize, cellsStart, out_xyzouvwo);
}

void buildCellList(float4* in_xyzouvwo, const int n, const float3 domainStart, const int3 ncells, const float invrc,
				   float4* out_xyzouvwo, uint8_t* cellsSize, int* cellsStart, cudaStream_t stream)
{
	// Compute cell sizes
	const int totcells = ncells.x*ncells.y*ncells.z;
	CUDA_CHECK( cudaMemsetAsync(cellsSize, 0, (totcells + 1)*sizeof(uint8_t), stream) );  // +1 to have correct cellsStart[totcells]
	computeCellsSizes<<< (n+127)/128, 128, 0, stream >>>(in_xyzouvwo, n, domainStart, ncells, invrc, (uint*)cellsSize);

	// Scan to get cell starts
	scan(cellsSize, totcells+1, cellsStart, stream);

	// Rearrange the data
	rearrangeParticles<<< (2*n+127)/128, 128, 0, stream >>>(in_xyzouvwo, n, domainStart, ncells, invrc, (uint*)cellsSize, cellsStart, out_xyzouvwo);
}
