#include "datatypes.h"
#include "scan.h"
#include "celllist.h"
#include "non_cached_rw.h"
#include "flows.h"

__global__ void blendStartSize(uchar4* cellsSize, int4* cellsStart, const int totcells)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (4*gid >= totcells) return;

	uchar4 sizes  = cellsSize [gid];

	cellsStart[gid] += make_int4(sizes.x << 26, sizes.y << 26, sizes.z << 26, sizes.w << 26);
}

template<typename Transform>
__global__ void computeCellSizesBeforeTimeIntegration(const float4* xyzouvwo, const float4* accs, const int n, const int nMovable,
		const float3 domainStart, const int3 ncells, const float invrc, const float dt, Transform transform, uint* cellsSize)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= n) return;

	float4 coo = xyzouvwo[gid*2];
	float4 vel = xyzouvwo[gid*2+1];
	float4 acc = accs[gid];
	transform(coo, vel, acc, dt, gid);

	int cid;
	if (gid < nMovable)
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

template<typename Transform>
__global__ void rearrangeAndIntegrate(const float4* in_xyzouvwo, const float4* accs, const int n, const int nMovable, const float3 domainStart, const int3 ncells,
		const float invrc, uint* cellsSize, int* cellsStart, const float dt, Transform transform, float4* out_xyzouvwo)
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
	float4 acc = accs[pid];


	// Send velocity to adjacent thread that has coordinate
	float4 othval;
	othval.x = __shfl_down(val.x, 1);
	othval.y = __shfl_down(val.y, 1);
	othval.z = __shfl_down(val.z, 1);
	othval.w = __shfl_down(val.w, 1);

	int cid = -1;
	if (sh == 0)
	{
		// val is coordinate, othval is corresponding velocity
		transform(val, othval, acc, dt, pid);

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
	{
		// val is velocity, othval is rubbish
		dstId = otherDst;
		transform(othval, val, acc, dt, pid);
	}

	if (dstId >= 0) writeNoCache(out_xyzouvwo + 2*dstId+sh, val);
}


void buildCellList(ParticleVector& pv, IniParser& config, cudaStream_t stream)
{
	buildCellListAndIntegrate(pv, config, 0, stream);
}

void buildCellListAndIntegrate(ParticleVector& pv, IniParser& config, float dt, cudaStream_t stream)
{
	// Look up the included file for explanation

	// Compute cell sizes
	debug("Computing cell sizes for %d particles with %d newcomers", pv.np, pv.received);
	CUDA_Check( cudaMemsetAsync(pv.cellsSize.devdata, 0, (pv.totcells + 1)*sizeof(uint8_t), stream) );  // +1 to have correct cellsStart[totcells]

	flowMacroWrapper( (computeCellSizesBeforeTimeIntegration<<< (pv.np+127)/128, 128, 0, stream >>> (
						(float4*)pv.coosvels.devdata, (float4*)pv.accs.devdata,
						pv.np, pv.np - pv.received, pv.domainStart, pv.ncells, 1.0f, dt, integrate, (uint*)pv.cellsSize.devdata)) );

	// Scan to get cell starts
	scan(pv.cellsSize.devdata, pv.totcells+1, pv.cellsStart.devdata, stream);

	// Blend size and start together
	blendStartSize<<< ((pv.totcells+3)/4 + 127) / 128, 128, 0, stream >>>((uchar4*)pv.cellsSize.devdata, (int4*)pv.cellsStart.devdata, pv.totcells);

	// Rearrange the data
	debug("Rearranging and integrating %d particles", pv.np);

	flowMacroWrapper( (rearrangeAndIntegrate<<< (2*pv.np+127)/128, 128, 0, stream >>> (
						(float4*)pv.coosvels.devdata, (float4*)pv.accs.devdata, pv.np, pv.np - pv.received, pv.domainStart, pv.ncells, 1.0f,
						(uint*)pv.cellsSize.devdata, pv.cellsStart.devdata, dt, integrate, (float4*)pv.pingPongBuf.devdata)) );

	debug("Rearranging completed");

	// Now we need the new size of particles array.
	int newSize;
	CUDA_Check( cudaMemcpyAsync(&newSize, pv.cellsStart.devdata + pv.totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
	CUDA_Check( cudaStreamSynchronize(stream) );
	pv.resize(newSize, resizePreserve, stream);

	swap(pv.coosvels, pv.pingPongBuf, stream);
}
