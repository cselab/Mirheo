#include "minmax-massimo.h"

#define MAXTHREADS 1024
#define WARPSIZE     32
#define MAXV     100000000.
#define MINV    -100000000.

__global__ void minmaxob(struct Particle *d_data, float3 *d_min, float3 *d_max, int size) {
    __shared__ float3 mintemp[32];
    __shared__ float3 maxtemp[32];
    __shared__ float shrtmp[3][MAXTHREADS];

    float3 mintemp1, maxtemp1;
    float3 mindef, maxdef;
    float temp2;
    if(blockDim.x>MAXTHREADS) {
        cuda_printf("Invalid number of threads per block: %d, must be <=%d\n",blockDim.x,MAXTHREADS);
    }
    mindef.x=MAXV;   mindef.y=MAXV;   mindef.z=MAXV;
    maxdef.x=MINV;   maxdef.y=MINV;   maxdef.z=MINV;
    __syncthreads();
    int tid = threadIdx.x;
    int xyz;
    for(int i=tid; i<3*blockDim.x; i+=blockDim.x) {
	xyz=i%3;
	//    if(xyz==0) {
	shrtmp[xyz][i/3] = (i/3<size)?d_data[i/3+blockIdx.x*size].x[xyz]:MINV;
	//    } else if(xyz==1) {
	//      shrtmp[xyz][i/3] = (i/3<size)?d_data[i/3+blockIdx.x*blockDim.x].y:MINV;
	//    } else {
	//      shrtmp[xyz][i/3] = (i/3<size)?d_data[i/3+blockIdx.x*blockDim.x].z:MINV;
	//    }
    }
    __syncthreads();
    mintemp1.x = (tid<size)?shrtmp[0][tid]:MAXV;
    mintemp1.y = (tid<size)?shrtmp[1][tid]:MAXV;
    mintemp1.z = (tid<size)?shrtmp[2][tid]:MAXV;
    maxtemp1.x = (tid<size)?shrtmp[0][tid]:MINV;
    maxtemp1.y = (tid<size)?shrtmp[1][tid]:MINV;
    maxtemp1.z = (tid<size)?shrtmp[2][tid]:MINV;
    for (int d=1; d<32; d<<=1) {
	temp2 = __shfl_up(mintemp1.x,d);
	mintemp1.x=(mintemp1.x>temp2)?temp2:mintemp1.x;
	temp2 = __shfl_up(mintemp1.y,d);
	mintemp1.y=(mintemp1.y>temp2)?temp2:mintemp1.y;
	temp2 = __shfl_up(mintemp1.z,d);
	mintemp1.z=(mintemp1.z>temp2)?temp2:mintemp1.z;
	temp2 = __shfl_up(maxtemp1.x,d);
	maxtemp1.x=(maxtemp1.x<temp2)?temp2:maxtemp1.x;
	temp2 = __shfl_up(maxtemp1.y,d);
	maxtemp1.y=(maxtemp1.y<temp2)?temp2:maxtemp1.y;
	temp2 = __shfl_up(maxtemp1.z,d);
	maxtemp1.z=(maxtemp1.z<temp2)?temp2:maxtemp1.z;
    }
    if (tid%32 == 31) {
	mintemp[tid/32] = mintemp1;
	maxtemp[tid/32] = maxtemp1;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        mintemp1= (tid < blockDim.x/32)?mintemp[threadIdx.x]:mindef;
        maxtemp1= (tid < blockDim.x/32)?maxtemp[threadIdx.x]:maxdef;
        for (int d=1; d<32; d<<=1) {
	    temp2 = __shfl_up(mintemp1.x,d);
	    mintemp1.x=(mintemp1.x>temp2)?temp2:mintemp1.x;
	    temp2 = __shfl_up(mintemp1.y,d);
	    mintemp1.y=(mintemp1.y>temp2)?temp2:mintemp1.y;
	    temp2 = __shfl_up(mintemp1.z,d);
	    mintemp1.z=(mintemp1.z>temp2)?temp2:mintemp1.z;
	    temp2 = __shfl_up(maxtemp1.x,d);
	    maxtemp1.x=(maxtemp1.x<temp2)?temp2:maxtemp1.x;
	    temp2 = __shfl_up(maxtemp1.y,d);
	    maxtemp1.y=(maxtemp1.y<temp2)?temp2:maxtemp1.y;
	    temp2 = __shfl_up(maxtemp1.z,d);
	    maxtemp1.z=(maxtemp1.z<temp2)?temp2:maxtemp1.z;
        }
        if (tid < blockDim.x/32) {
	    mintemp[tid] = mintemp1;
	    maxtemp[tid] = maxtemp1;
        }
    }
    __syncthreads();
    if (threadIdx.x==blockDim.x-1) {
	d_min[blockIdx.x]=mintemp[blockDim.x/32-1];
	d_max[blockIdx.x]=maxtemp[blockDim.x/32-1];
    }

}

void minmax_massimo(Particle * rbc, int size, int n, float3 *minrbc, float3 *maxrbc) 
{
    minmaxob<<<n,((size+31)/32)*32>>>(rbc, minrbc, maxrbc, size);
}