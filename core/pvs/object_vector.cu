#include "object_vector.h"

__global__ void min_max_com(const float4 * coosvels, LocalObjectVector::COMandExtent* com_ext, const int nObj, const int objSize)
{
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	const int objId = gid >> 5;
	const int tid = gid & 0x1f;
	if (objId >= nObj) return;

	float3 mymin = make_float3( 1e+10f);
	float3 mymax = make_float3(-1e+10f);
	float3 mycom = make_float3(0);

#pragma unroll 3
	for (int i = tid; i < objSize; i += warpSize)
	{
		const int offset = (objId * objSize + i) * 2;

		const float3 coo = make_float3(coosvels[offset]);

		mymin = fminf(mymin, coo);
		mymax = fmaxf(mymax, coo);
		mycom += coo;
	}

	mycom = warpReduce( mycom, [] (float a, float b) { return a+b; } );
	mymin = warpReduce( mymin, [] (float a, float b) { return fmin(a, b); } );
	mymax = warpReduce( mymax, [] (float a, float b) { return fmax(a, b); } );

	if (tid == 0)
		com_ext[objId] = {mycom / objSize, mymin, mymax};
}

void LocalObjectVector::findExtentAndCOM(cudaStream_t stream)
{
	const int nthreads = 128;
	min_max_com<<< (nObjects*32 + nthreads-1)/nthreads, nthreads, 0, stream >>> ((float4*)coosvels.devPtr(), comAndExtents.devPtr(), nObjects, objSize);
}
