#include <core/object_vector.h>
#include <core/helper_math.h>

template<typename Operation>
__inline__ __device__ float3 warpReduce(float3 val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = op(val.x, __shfl_down(val.x, offset));
		val.y = op(val.y, __shfl_down(val.y, offset));
		val.z = op(val.z, __shfl_down(val.z, offset));
	}
	return val;
}

__global__ void min_max_com(const float4 * coosvels, ObjectVector::Properties* props, const int nObj, const int objSize)
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
		props[objId] = {mycom / objSize, mymin, mymax};
}

void ObjectVector::findExtentAndCOM(cudaStream_t stream)
{
	const int nthreads = 128;
	min_max_com<<< (nObjects*32 + nthreads-1)/nthreads, nthreads, 0, stream >>> ((float4*)coosvels.devPtr(), properties.devPtr(), nObjects, objSize);
}
