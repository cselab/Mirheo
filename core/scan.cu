#include <core/scan.h>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/cuda_common.h>

//#include <cstdio>


__device__ __forceinline__ void int2_4ints(const uint v, int res[4])
{
	res[3] = v >> 24;
	res[2] = (v & 0x00ff0000) >> 16;
	res[1] = (v & 0x0000ff00) >> 8;
	res[0] = v & 0x000000ff;
}

template<int NWARPS>
__global__ void scanCharLocal(const uint4* in, const int n, int* blockScan)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if (gid*16 >= n) return;

	const int tid = threadIdx.x & 31;
	const int wid = threadIdx.x >> 5;

	volatile __shared__ int partScan[NWARPS];

	const uint4 tmp = in[gid];
	int v[16];

	int2_4ints(tmp.x, v);
	int2_4ints(tmp.y, v+4);
	int2_4ints(tmp.z, v+8);
	int2_4ints(tmp.w, v+12);

#pragma unroll
	for (int i=1; i<16; i++)
		v[i] = (gid*16 + i < n) ? v[i] + v[i-1] : v[i-1];

	int myscan = v[15];
#pragma unroll
	for (int sh=1; sh<32; sh = sh*2)
	{
		const int othscan =  __shfl_up(myscan, sh);
		if (tid >= sh) myscan += othscan;
	}

	if (tid == 31) partScan[wid] = myscan;

	__syncthreads();

	if (threadIdx.x == blockDim.x-1)
	{
		for (int i=0; i<wid; i++)
			myscan += partScan[i];
		blockScan[blockIdx.x] = myscan;
	}
}

template<int NWARPS>
__global__ void scanBlock(int4* inout, const int n)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if (gid*4 > n) return;

	const int tid = threadIdx.x & 31;
	const int wid = threadIdx.x >> 5;

	volatile __shared__ int partScan[NWARPS];

	const int4 tmp = inout[gid];
	int v[4] = {tmp.x, tmp.y, tmp.z, tmp.w};

#pragma unroll
	for (int i=1; i<4; i++)
		v[i] = (gid*4 + i < n) ? v[i] + v[i-1] : v[i-1];

	int myscan = v[3];

#pragma unroll
	for (int sh=1; sh<32; sh = sh*2)
	{
		const int othscan = __shfl_up(myscan, sh);
		if (tid >= sh) myscan += othscan;
	}

	if (tid == 31) partScan[wid] = myscan;
	myscan -= v[3];

	__syncthreads();

#pragma unroll 2
	for (int i=0; i<wid; i++)
		myscan += partScan[i];

	inout[gid] = make_int4(myscan) + make_int4(0, v[0], v[1], v[2]);  // exclusive scan
}


template<int NWARPS>
__global__ void addBlockScan(const uchar4* in, const int n, const int* blockScan, int4* out)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if (gid*4 >= n) return;

	const int tid = threadIdx.x & 31;
	const int wid = threadIdx.x >> 5;

	volatile __shared__ int partScan[NWARPS];

	const uchar4 tmp = in[gid];
	uint v[4] = {tmp.x, tmp.y, tmp.z, tmp.w};

#pragma unroll
	for (int i=1; i<4; i++)
		v[i] = (gid*4 + i < n) ? v[i] + v[i-1] : v[i-1];

	int myscan = v[3];
#pragma unroll
	for (int sh=1; sh<32; sh = sh*2)
	{
		const int othscan =  __shfl_up(myscan, sh);
		if (tid >= sh) myscan += othscan;
	}

	if (tid == 31) partScan[wid] = myscan;
	myscan += blockScan[blockIdx.x] - v[3];

	__syncthreads();

	for (int i=0; i<wid; i++)
		myscan += partScan[i];

	const int4 res = make_int4( 0, v[0], v[1], v[2]) + make_int4(myscan);
	out[gid] = res;
}

void scan(const uint8_t* in, const int n, int* out, cudaStream_t stream)
{
	assert(n <= 16*32*16 * 32*32);

	const int initialWarps   = 4;
	const int initialThreads = initialWarps * 32;
	const int initialBlocks  = ((n+15) / 16 + initialThreads - 1) / initialThreads;

	static DeviceBuffer<int> blockScan;
	blockScan.resize( n / (initialWarps*32*16) + 1, stream );

	const int blockWarps     = 16;
	const int blockThreads   = blockWarps * 32;
	const int blockBlocks    = ((blockScan.size()+3) / 4 + blockThreads - 1) / blockThreads;

	const int globWarps      = initialWarps * 4;
	const int globThreads    = blockWarps * 32;
	const int globBlocks     = ((n+3) / 4 + blockThreads - 1) / blockThreads;

	scanCharLocal<initialWarps> <<< initialBlocks, initialThreads, 0, stream >>>
			((uint4*)in, n, blockScan.devPtr());

	scanBlock<blockWarps>       <<< blockBlocks,   blockThreads,   0, stream >>>
			((int4*)blockScan.devPtr(), blockScan.size());

	addBlockScan<globWarps>  <<< globBlocks, globThreads, 0, stream >>>
			((uchar4*)in, n, blockScan.devPtr(), (int4*)out);
}
