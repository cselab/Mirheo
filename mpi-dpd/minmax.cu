/*
 *  minmax-massimo.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Massimo Bernaschi on 2015-03-23.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include "minmax.h"

#define MAXTHREADS 1024
#define WARPSIZE     32
#define MAXV     100000000.
#define MINV    -100000000.

typedef struct 
{
    int g_block_id;
    int g_blockcnt;
    float3 minval;
    float3 maxval;
} sblockds_t;

__global__ void minmaxob(const Particle * const d_data, float3 *d_min, float3 *d_max, int size) {
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


__global__ void minmaxmba(const Particle  *d_data, float3 *d_min, float3 *d_max,
			  int size, sblockds_t *ptoblockds) {

  __shared__ float3 mintemp[32];
  __shared__ float3 maxtemp[32];
  __shared__ float shrtmp[3][MAXTHREADS];

  __shared__ unsigned int my_blockId;
  const int which=blockIdx.x/((size+blockDim.x-1)/blockDim.x); /* which particle should manage */
  float3 mintemp1, maxtemp1;
  float3 mindef, maxdef;
  float temp2;
  if(blockDim.x > MAXTHREADS) {
        cuda_printf("Invalid number of threads per block: %d, must be <=%d\n",blockDim.x,MAXTHREADS);
  }
  if (threadIdx.x==0) {
    my_blockId = atomicAdd( &(ptoblockds[which].g_block_id), 1 );
  }
  mindef.x=MAXV;   mindef.y=MAXV;   mindef.z=MAXV;
  maxdef.x=MINV;   maxdef.y=MINV;   maxdef.z=MINV;
  __syncthreads();
  int tid = threadIdx.x;
  int xyz;
  for(int i=tid; i<3*blockDim.x; i+=blockDim.x) {
    xyz=i%3;
    //    if(xyz==0) {
    shrtmp[xyz][i/3] = (i/3+my_blockId*blockDim.x<size)?d_data[i/3+my_blockId*blockDim.x+which*size].x[xyz]:MINV;
    //    } else if(xyz==1) {
    //      shrtmp[xyz][i/3] = (i/3+my_blockId*blockDim.x<size)?d_data[i/3+my_blockId*blockDim.x+which*size].y:MINV;
    //    } else {
    //      shrtmp[xyz][i/3] = (i/3+my_blockId*blockDim.x<size)?d_data[i/3+my_blockId*blockDim.x+which*size].z:MINV;
     //    }
  }
  __syncthreads();
  mintemp1.x = (tid+my_blockId*blockDim.x<size)?shrtmp[0][tid]:MAXV;
  mintemp1.y = (tid+my_blockId*blockDim.x<size)?shrtmp[1][tid]:MAXV;
  mintemp1.z = (tid+my_blockId*blockDim.x<size)?shrtmp[2][tid]:MAXV;
  maxtemp1.x = (tid+my_blockId*blockDim.x<size)?shrtmp[0][tid]:MINV;
  maxtemp1.y = (tid+my_blockId*blockDim.x<size)?shrtmp[1][tid]:MINV;
  maxtemp1.z = (tid+my_blockId*blockDim.x<size)?shrtmp[2][tid]:MINV;
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
  mintemp1=mintemp[blockDim.x/32-1];
  maxtemp1=maxtemp[blockDim.x/32-1];
  if (threadIdx.x==(blockDim.x-1)) {
        do {} while( atomicAdd(&(ptoblockds[which].g_blockcnt),0) < my_blockId );
        mintemp1.x=(ptoblockds[which].minval.x<mintemp1.x)?ptoblockds[which].minval.x:mintemp1.x;
        maxtemp1.x=(ptoblockds[which].maxval.x>maxtemp1.x)?ptoblockds[which].maxval.x:maxtemp1.x;
        mintemp1.y=(ptoblockds[which].minval.y<mintemp1.y)?ptoblockds[which].minval.y:mintemp1.y;
        maxtemp1.y=(ptoblockds[which].maxval.y>maxtemp1.y)?ptoblockds[which].maxval.y:maxtemp1.y;
        mintemp1.z=(ptoblockds[which].minval.z<mintemp1.z)?ptoblockds[which].minval.z:mintemp1.z;
        maxtemp1.z=(ptoblockds[which].maxval.z>maxtemp1.z)?ptoblockds[which].maxval.z:maxtemp1.z;
        if(my_blockId==(((size+blockDim.x-1)/blockDim.x))-1) { /* it is the last block; reset for next iteration */
                ptoblockds[which].minval=mindef;
                ptoblockds[which].maxval=maxdef;
                ptoblockds[which].g_blockcnt=0;
                ptoblockds[which].g_block_id=0;
                d_min[which]=mintemp1;
                d_max[which]=maxtemp1;
        } else {
                ptoblockds[which].minval=mintemp1;
                ptoblockds[which].maxval=maxtemp1;
                atomicAdd(&(ptoblockds[which].g_blockcnt),1);
        }
  }

}

void minmax(const Particle * const rbc, int size, int n, float3 *minrbc, float3 *maxrbc, cudaStream_t stream) 
{
    const int size32 = ((size + 31) / 32) * 32;

    if (size32 < MAXTHREADS)
	minmaxob<<<n, size32, 0, stream>>>(rbc, minrbc, maxrbc, size);
    else
    {
	static int nctc = -1;

        static sblockds_t *ptoblockds = NULL;
	
        if( n > nctc) 
	{
	    sblockds_t * h_ptoblockds = new sblockds_t[n];
	    
	    for(int i=0; i < n; i++)  
	    {
		h_ptoblockds[i].g_block_id=0;
		h_ptoblockds[i].g_blockcnt=0;
		h_ptoblockds[i].minval.x=MAXV;
		h_ptoblockds[i].maxval.x=MINV;
		h_ptoblockds[i].minval.y=MAXV;
		h_ptoblockds[i].maxval.y=MINV;
		h_ptoblockds[i].minval.z=MAXV;
		h_ptoblockds[i].maxval.z=MINV;
	    }

	    if (ptoblockds != NULL) 
		CUDA_CHECK(cudaFree(ptoblockds));

           CUDA_CHECK(cudaMalloc((void **)&ptoblockds,sizeof(sblockds_t) * n));

           CUDA_CHECK(cudaMemcpy(ptoblockds, h_ptoblockds, sizeof(sblockds_t) * n, cudaMemcpyHostToDevice));

           delete [] h_ptoblockds;
        }

        int nblocks= n * ((size + MAXTHREADS - 1) / MAXTHREADS);

        minmaxmba<<<nblocks, MAXTHREADS, 0, stream>>>(rbc, minrbc, maxrbc, size, ptoblockds);
    }
}