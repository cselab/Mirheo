#include <core/scan.h>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/cuda_common.h>

__device__ inline void wscan(int* in, int* out, int lid, int stride = 1, int prefix=0)
{
	int myv = in[lid*stride];

#pragma unroll
	for (int p=1; p<=16; p <<= 1)
	{
		int ov = __shfl_up(myv, p);
		if (lid >= p) myv += ov;
	}

	out[lid] = myv + prefix;
}

__device__ __forceinline__ int4 int2_4ints(const uint v)
{
	int4 res;
	res.w = v >> 24;
	res.z = (v & 0x00ff0000) >> 16;
	res.y = (v & 0x0000ff00) >> 8;
	res.x = v & 0x000000ff;

	return res;
}


template<int CHSIZE, int NWARPS>
__global__ void _scan(const uint4 *in, int4* out, int n, int* curChunk, int* partScan, int* nSums)
{
	const int tid = threadIdx.x;
	const int wid = tid / warpSize;
	const int lid = tid % warpSize;

	const int nWarpScans = 4*CHSIZE / 32;

	__shared__ int chId;
	__shared__ int4 b4[CHSIZE];
	__shared__ int warpScan[nWarpScans];
	int* buffer = (int*)b4;

	if (tid == 0)
		chId = atomicAdd(curChunk, 1);

	__syncthreads();

	int start = chId * CHSIZE/4;
	int end = min(start + CHSIZE/4, n/16);

	//for (int i=start+tid; i<end; i+=blockDim.x)
	//{
		uint4 v = in[tid+start];
		b4[ 4*(tid)+0 ] = int2_4ints(v.x);
		b4[ 4*(tid)+1 ] = int2_4ints(v.y);
		b4[ 4*(tid)+2 ] = int2_4ints(v.z);
		b4[ 4*(tid)+3 ] = int2_4ints(v.w);
	//}

	__syncthreads();

#pragma unroll 4
	for (int i=wid*warpSize; i < 4*CHSIZE; i+=blockDim.x)
		wscan(buffer + i, buffer + i, lid);

	__syncthreads();

	if (wid == 0)
	{
#pragma unroll 2
		for (int i=0; i < nWarpScans; i+=warpSize)
			wscan(buffer + i*warpSize + 31, warpScan + i, lid, warpSize, (i < warpSize) ? 0 : warpScan[i - 1]);
	}

	__syncthreads();

	int nChunks = n / (4*CHSIZE) + 1;

	for (int i=chId+1 + tid; i<nChunks; i+=blockDim.x)
	{
		atomicAdd(partScan+i, warpScan[nWarpScans - 1]);
		atomicAdd(nSums + i, 1);
	}

	volatile int* myAddr = nSums + chId;
	if (chId > 0)
		while (*myAddr != chId);

	int myPrefix = (chId == 0) ? 0 : partScan[chId];

	int* in1 = (int*) in;

#pragma unroll 4
	for (int i=tid; i < CHSIZE; i+=blockDim.x)
	{
		int id = (4*i)/warpSize - 1;
		int tmp = myPrefix + ((id >= 0) ? warpScan[id] : 0);

		out[chId*CHSIZE + i] = make_int4(
				buffer[4*i+0] + tmp,
				buffer[4*i+1] + tmp,
				buffer[4*i+2] + tmp,
				buffer[4*i+3] + tmp) - int2_4ints(in1[chId*CHSIZE + i]);
	}
}




void scan(const uint8_t* in, const int n, int* out, cudaStream_t stream)
{
	assert(n <= 16*32*16 * 32*32);

	const int chSize = 1024;
	const int nwarps = 8;
	const int nthreads = nwarps * 32;
	const int nblocks = getNblocks(n, 4*chSize);

	static DeviceBuffer<int> curChunk, partScan, nSums;

	curChunk.resize(1, stream);
	partScan.resize(n/nthreads + 1, stream);
	nSums.resize(n/nthreads + 1, stream);

	curChunk.clear(stream);
	partScan.clear(stream);
	nSums.clear(stream);

	_scan<chSize, nwarps> <<< nblocks, nthreads, 0, stream >>> (
			(uint4*)in, (int4*)out, n, curChunk.devPtr(), partScan.devPtr(), nSums.devPtr());

}
