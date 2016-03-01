/*
 *  velcontroller.cu
 *  ctc falcon
 *
 *  Created by Dmitry Alexeev on Sep 24, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#include "velcontroller.h"
#include "helper_math.h"

//====================================================================================
// Kernels
//====================================================================================

namespace VelContKernels
{
    __global__ void sample(const int * const __restrict__ cellsstart, const float2* const __restrict__ p, float3* res, VelController::CellInfo info)
    {
        const uint3 ccoos = {threadIdx.x + blockIdx.x*blockDim.x,
                threadIdx.y + blockIdx.y*blockDim.y,
                threadIdx.z + blockIdx.z*blockDim.z};

        if (ccoos.x < info.n[0] && ccoos.y < info.n[1] && ccoos.z < info.n[2])
        {
            const uint cid = (ccoos.x + info.xl[0]) + (ccoos.y + info.xl[1]) * info.cellsx + (ccoos.z + info.xl[2]) * info.cellsx * info.cellsy;
            const uint resid = ccoos.x + ccoos.y * info.n[0] + ccoos.z * info.n[0] * info.n[1];

            float3 myres = make_float3(0.0f, 0.0f, 0.0f);
            for (uint pid = cellsstart[cid]; pid < cellsstart[cid+1]; pid++)
            {
                float num = cellsstart[cid+1] - cellsstart[cid];
                float2 tmp1 = p[3*pid + 1];
                float2 tmp2 = p[3*pid + 2];
                myres.x += tmp1.y / num;
                myres.y += tmp2.x / num;
                myres.z += tmp2.y / num;
            }
            res[resid] += make_float3(myres.x, myres.y, myres.z);
        }
    }

    __global__ void push(const int * const __restrict__ cellsstart, Acceleration* acc, float3 f, VelController::CellInfo info)
    {
        const uint3 ccoos = {threadIdx.x + blockIdx.x*blockDim.x,
                threadIdx.y + blockIdx.y*blockDim.y,
                threadIdx.z + blockIdx.z*blockDim.z};

        if (ccoos.x < info.n[0] && ccoos.y < info.n[1] && ccoos.z < info.n[2])
        {
            const uint cid = (ccoos.x + info.xl[0]) + (ccoos.y + info.xl[1]) * info.cellsx + (ccoos.z + info.xl[2]) * info.cellsx * info.cellsy;

            for (uint pid = cellsstart[cid]; pid < cellsstart[cid+1]; pid++)
            {
                acc[pid].a[0] += f.x;
                //acc[pid].a[1] += f.y;
                //acc[pid].a[2] += f.z;
            }
        }
    }

    __inline__ __device__ float3 warpReduceSum(float3 val)
    {
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            val.x += __shfl_down(val.x, offset);
            val.y += __shfl_down(val.y, offset);
            val.z += __shfl_down(val.z, offset);
        }
        return val;
    }

    __global__ void reduceByWarp(float3 *res, const float3 * const __restrict__ vel, const uint total)
    {
        assert(blockDim.x == 32);
        const uint id = threadIdx.x + blockIdx.x*blockDim.x;
        const uint ch = blockIdx.x;
        if (id >= total) return;

        const float3 val  = vel[id];
        const float3 rval = warpReduceSum(val);

        if ((threadIdx.x % warpSize) == 0)
            res[ch]=rval;
    }
}

//====================================================================================
// Methods
//====================================================================================

VelController::VelController(int xl[3], int xh[3], int mpicoos[3], float3 desired, MPI_Comm comm) :
                                desired(desired), Kp(2), Ki(1), Kd(8), factor(0.01), sampleid(0)
{
    MPI_CHECK( MPI_Comm_dup(comm, &this->comm) );
    MPI_CHECK( MPI_Comm_size(comm, &size) );
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );
    const int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};

    int myxl[3], myxh[3], n[3];
    for (int d=0; d<3; d++)
    {
        myxl[d] = max(xl[d], L[d]*mpicoos[d]    );
        myxh[d] = min(xh[d], L[d]*(mpicoos[d]+1));

        info.n[d] = n[d] = myxh[d] - myxl[d];
        info.xl[d] = myxl[d] % (L[d]+1);
    }

    if (n[0] > 0 && n[1] > 0 && n[2] > 0)
        total = n[0] * n[1] * n[2];
    else
        total = 0;

    MPI_CHECK( MPI_Allreduce(&total, &globtot, 1, MPI_INT, MPI_SUM, comm) );

    vel.resize(total);
    if (total)
        CUDA_CHECK( cudaMemset(vel.data, 0, n[0] * n[1] * n[2] * sizeof(float3)) );

    info.cellsx = L[0];
    info.cellsy = L[1];
    info.cellsz = L[2];

    s = f = make_float3(0, 0, 0);
    old = desired;
}

void VelController::sample(const int * const cellsstart, const Particle* const p, cudaStream_t stream)
{
    dim3 block(8, 8, 1);
    dim3 grid(  (info.n[0] + block.x - 1) / block.x,
            (info.n[1] + block.y - 1) / block.y,
            (info.n[2] + block.z - 1) / block.z );

    sampleid++;
    if (total)
        VelContKernels::sample <<<grid, block, 0, stream>>> (cellsstart, (float2*)p, vel.data, info);
}

void VelController::push(const int * const cellsstart, const Particle* const p, Acceleration* acc, cudaStream_t stream)
{
    dim3 block(8, 8, 1);
    dim3 grid(  (info.n[0] + block.x - 1) / block.x,
            (info.n[1] + block.y - 1) / block.y,
            (info.n[2] + block.z - 1) / block.z );

    if (total)
        VelContKernels::push <<<grid, block, 0, stream>>> (cellsstart, acc, f, info);
}

float3 VelController::adjustF(cudaStream_t stream)
{
    const int chunks = (total+31) / 32;
    if (avgvel.size < chunks) avgvel.resize(chunks);

    if (total)
    {
        VelContKernels::reduceByWarp <<< (total + 31) / 32, 32, 0, stream >>> (avgvel.devptr, vel.data, total);
        CUDA_CHECK( cudaStreamSynchronize(stream) );
    }

    float3 cur = make_float3(0, 0, 0);
    for (int i=0; i<chunks; i++)
        cur += avgvel.data[i];

    MPI_CHECK( MPI_Allreduce(MPI_IN_PLACE, &cur.x, 3, MPI_FLOAT, MPI_SUM, comm) );
    cur /= globtot * sampleid;

    float3 err = desired - cur;
    float3 de  = err - old;
    s += err;
    f = factor*(Kp*err + Ki*s + Kd*de);

    if (total)
        CUDA_CHECK( cudaMemsetAsync(vel.data, 0, total * sizeof(float3), stream) );
    sampleid = 0;

    if (rank==0) printf("Vel:  [%8.3f  %8.3f  %8.3f], force: [%8.3f  %8.3f  %8.3f]\n", cur.x, cur.y, cur.z,  f.x, f.y, f.z);

    return f;
}

