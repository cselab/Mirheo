#pragma once

#include "../average_flow.h"

#include <core/utils/cuda_common.h>
#include <core/pvs/particle_vector.h>

struct ChannelsInfo
{
    int n;
    Average3D::ChannelType *types;
    float **average, **data;

    ChannelsInfo(Average3D::HostChannelsInfo& info, ParticleVector* pv, cudaStream_t stream)
    {
        for (int i = 0; i < info.n; i++)
        {
            if (info.names[i] == "velocity") info.dataPtrs[i] = (float*) pv->local()->velocities().devPtr();
            else info.dataPtrs[i] = (float*)pv->local()->dataPerParticle.getGenericPtr(info.names[i]);
        }

        info.dataPtrs.uploadToDevice(stream);
        CUDA_Check( cudaStreamSynchronize(stream) );

        n = info.n;
        types = info.types.devPtr();
        average = info.averagePtrs.devPtr();
        data = info.dataPtrs.devPtr();
    }
};


namespace SamplingHelpersKernels
{

__device__ inline void sampleChannels(int pid, int cid, ChannelsInfo channelsInfo)
{
    for (int i=0; i<channelsInfo.n; i++)
    {
        if (channelsInfo.types[i] == Average3D::ChannelType::Scalar)
            atomicAdd(channelsInfo.average[i] + cid, channelsInfo.data[i][pid]);

        if (channelsInfo.types[i] == Average3D::ChannelType::Vector_float3)
            atomicAdd(((float3*)channelsInfo.average[i]) + cid, ((float3*)channelsInfo.data[i])[pid]);

        if (channelsInfo.types[i] == Average3D::ChannelType::Vector_float4)
            atomicAdd(((float3*)channelsInfo.average[i]) + cid, make_float3( ((float4*)channelsInfo.data[i])[pid] ));

        if (channelsInfo.types[i] == Average3D::ChannelType::Vector_2xfloat4)
            atomicAdd(((float3*)channelsInfo.average[i]) + cid, make_float3( ((float4*)channelsInfo.data[i])[2*pid] ));

        if (channelsInfo.types[i] == Average3D::ChannelType::Tensor6)
        {
            atomicAdd(((float3*)channelsInfo.average[i]) + 2*cid + 0, ((float3*)channelsInfo.data[i])[2*pid + 0] );
            atomicAdd(((float3*)channelsInfo.average[i]) + 2*cid + 1, ((float3*)channelsInfo.data[i])[2*pid + 1] );
        }
    }
}

__global__ static void scaleVec(int n, int fieldComponents, double *field, const double *density)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n)
        for (int c=0; c<fieldComponents; c++)
            if (fabs(density[id]) > 1e-6f)
                field[fieldComponents*id + c] /= density[id];
            else
                field[fieldComponents*id + c] = 0.0f;
}

__global__ static void correctVelocity(int n, double3 *velocity, const double *density, const float3 correction)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id >= n) return;

    velocity[id].x -= density[id] * correction.x;
    velocity[id].y -= density[id] * correction.y;
    velocity[id].z -= density[id] * correction.z;
}

__global__ static void scaleDensity(int n, double *density, const float factor)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n)
        density[id] *= factor;
}

__global__ static void accumulate(int n, int fieldComponents, const float *src, double *dst)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n * fieldComponents)
        dst[id] += src[id];
}

} // namespace SamplingHelpersKernels
