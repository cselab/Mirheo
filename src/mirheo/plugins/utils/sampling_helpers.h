// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "../average_flow.h"

#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

template <typename T>
inline bool checkType(__UNUSED const Average3D::ChannelType& channelType) { return false;}

template <> inline bool checkType<real> (const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Scalar;}
template <> inline bool checkType<real3>(const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Vector_real3;}
template <> inline bool checkType<real4>(const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Vector_real4;}
template <> inline bool checkType<Stress>(const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Tensor6;}
template <> inline bool checkType<Force> (const Average3D::ChannelType& channelType) { return checkType<real4> (channelType);}

static real* getDataAndCheck(const std::string& name, LocalParticleVector *lpv, const Average3D::ChannelType& channelType)
{
    const auto& contDesc = lpv->dataPerParticle.getChannelDescOrDie(name);

    return mpark::visit([&](auto pinnedBuff)
    {
        using T = typename std::remove_reference< decltype(*pinnedBuff->hostPtr()) >::type;
        if (!checkType<T>(channelType))
            die("incompatible type for channel '%s'", name.c_str());
        return (real*) pinnedBuff->devPtr();
    }, contDesc.varDataPtr);
}

struct ChannelsInfo
{
    int n;
    Average3D::ChannelType *types;
    real **average, **data;

    ChannelsInfo(Average3D::HostChannelsInfo& info, ParticleVector *pv, cudaStream_t stream)
    {
        for (int i = 0; i < info.n; i++)
            info.dataPtrs[i] = getDataAndCheck(info.names[i], pv->local(), info.types[i]);

        info.dataPtrs.uploadToDevice(stream);
        CUDA_Check( cudaStreamSynchronize(stream) );

        n       = info.n;
        types   = info.types.devPtr();
        average = info.averagePtrs.devPtr();
        data    = info.dataPtrs.devPtr();
    }
};


namespace sampling_helpers_kernels
{

__device__ inline void sampleChannels(int pid, int cid, ChannelsInfo channelsInfo)
{
    for (int i = 0; i < channelsInfo.n; ++i)
    {
        if (channelsInfo.types[i] == Average3D::ChannelType::Scalar)
            atomicAdd(channelsInfo.average[i] + cid, channelsInfo.data[i][pid]);

        if (channelsInfo.types[i] == Average3D::ChannelType::Vector_real3)
            atomicAdd(((real3*)channelsInfo.average[i]) + cid, ((real3*)channelsInfo.data[i])[pid]);

        if (channelsInfo.types[i] == Average3D::ChannelType::Vector_real4)
            atomicAdd(((real3*)channelsInfo.average[i]) + cid, make_real3( ((real4*)channelsInfo.data[i])[pid] ));

        if (channelsInfo.types[i] == Average3D::ChannelType::Tensor6)
        {
            atomicAdd(((real3*)channelsInfo.average[i]) + 2*cid + 0, ((real3*)channelsInfo.data[i])[2*pid + 0] );
            atomicAdd(((real3*)channelsInfo.average[i]) + 2*cid + 1, ((real3*)channelsInfo.data[i])[2*pid + 1] );
        }
    }
}

__global__ static void scaleVec(int n, int fieldComponents, double *field, const double *numberDensity)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id >= n) return;

    const double nd = numberDensity[id];

    for (int c = 0; c < fieldComponents; ++c)
        if (math::abs(nd) > 1e-6_r)
            field[fieldComponents*id + c] /= nd;
        else
            field[fieldComponents*id + c] = 0.0_r;
}

__global__ static void correctVelocity(int n, double3 *velocity, const double *numberDensity, const real3 correction)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id >= n) return;

    const double nd = numberDensity[id];

    velocity[id].x -= nd * correction.x;
    velocity[id].y -= nd * correction.y;
    velocity[id].z -= nd * correction.z;
}

__global__ static void scaleDensity(int n, double *numberDensity, const real factor)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n)
        numberDensity[id] *= factor;
}

__global__ static void accumulate(int n, int fieldComponents, const real *src, double *dst)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n * fieldComponents)
        dst[id] += src[id];
}

} // namespace sampling_helpers_kernels

} // namespace mirheo
