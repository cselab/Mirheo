/*
 *  velsampler.cu
 *  ctc PANDA
 *
 *  Created by Dmitry Alexeev on Nov 27, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "velsampler.h"
#include "helper_math.h"


//====================================================================================
// Kernels
//====================================================================================

namespace VelSmpKernels
{
    __global__ void sample(const int * const __restrict__ cellsstart, const float2* const __restrict__ p, float3* res, VelSampler::CellInfo info)
    {
        const uint3 ccoos = {threadIdx.x + blockIdx.x*blockDim.x,
                threadIdx.y + blockIdx.y*blockDim.y,
                threadIdx.z + blockIdx.z*blockDim.z};

        if (ccoos.x < info.cellsx && ccoos.y < info.cellsy && ccoos.z < info.cellsz)
        {
            const uint cid = ccoos.x + (ccoos.y + ccoos.z * info.cellsy) * info.cellsx;

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
            res[cid] += make_float3(myres.x, myres.y, myres.z);
        }
    }

    __global__ void scale(int n, float a, float* res)
    {
        const uint id = threadIdx.x + blockIdx.x*blockDim.x;

        if (id < n)
        {
            res[id] *= a;
        }
    }
}

//====================================================================================
// Methods
//====================================================================================


VelSampler::VelSampler()
{
    const int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};

    vels.resize(L[0]*L[1]*L[2]);
    hostVels.resize(vels.size);
    CUDA_CHECK( cudaMemset(vels.data, 0, vels.size * sizeof(float3)) );

    info.cellsx = L[0];
    info.cellsy = L[1];
    info.cellsz = L[2];
}

void VelSampler::sample(const int * const cellsstart, const Particle* const p, cudaStream_t stream)
{
    dim3 block(4, 4, 4);
    dim3 grid(  (info.cellsx + block.x - 1) / block.x,
            (info.cellsy + block.y - 1) / block.y,
            (info.cellsz + block.z - 1) / block.z );

    sampleid++;
    VelSmpKernels::sample <<<grid, block, 0, stream>>> (cellsstart, (float2*)p, vels.data, info);
    CUDA_CHECK(cudaPeekAtLastError());
}

vector<float3>& VelSampler::getAvgVel(cudaStream_t stream)
{
    if (sampleid > 0)
        VelSmpKernels::scale <<<(3*vels.size + 127) / 128, 128, 0, stream>>> (3*vels.size, 1.0f / sampleid, (float*)vels.data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpyAsync(&hostVels[0], vels.data, sizeof(float3) * vels.size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemsetAsync(vels.data, 0, sizeof(float3) * vels.size, stream));

    sampleid = 0;
    return hostVels;
}


