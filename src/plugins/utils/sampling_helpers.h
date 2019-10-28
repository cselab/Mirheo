#pragma once

#include "../average_flow.h"

#include <core/utils/cuda_common.h>
#include <core/pvs/particle_vector.h>

template <typename T>
inline bool checkType(__UNUSED const Average3D::ChannelType& channelType) { return false;}

template <> inline bool checkType<real> (const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Scalar;}
template <> inline bool checkType<real3>(const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Vector_real3;}
template <> inline bool checkType<real4>(const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Vector_real4;}
template <> inline bool checkType<Stress>(const Average3D::ChannelType& channelType) { return channelType == Average3D::ChannelType::Tensor6;}
template <> inline bool checkType<Force> (const Average3D::ChannelType& channelType) { return checkType<real4> (channelType);;}

static real* getDataAndCheck(const std::string& name, LocalParticleVector *lpv, const Average3D::ChannelType& channelType)
{
    if (name == "velocity") {
        if (channelType != Average3D::ChannelType::Vector_real4)
            die("incompatible type for channel '%s'", name.c_str());
        return (real*) lpv->velocities().devPtr();
    }
    else
    {
        const auto& contDesc = lpv->dataPerParticle.getChannelDescOrDie(name);

        return mpark::visit([&](auto pinnedBuff) {
            using T = typename std::remove_reference< decltype(*pinnedBuff->hostPtr()) >::type;
            if (!checkType<T>(channelType))
                die("incompatible type for channel '%s'", name.c_str());
            return (real*) pinnedBuff->devPtr();
        }, contDesc.varDataPtr);
    }
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


namespace SamplingHelpersKernels
{

__device__ inline void sampleChannels(int pid, int cid, ChannelsInfo channelsInfo)
{
    for (int i=0; i<channelsInfo.n; i++)
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

__global__ static void scaleVec(int n, int fieldComponents, double *field, const double *density)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n)
        for (int c = 0; c < fieldComponents; ++c)
            if (math::abs(density[id]) > 1e-6_r)
                field[fieldComponents*id + c] /= density[id];
            else
                field[fieldComponents*id + c] = 0.0_r;
}

__global__ static void correctVelocity(int n, double3 *velocity, const double *density, const real3 correction)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id >= n) return;

    velocity[id].x -= density[id] * correction.x;
    velocity[id].y -= density[id] * correction.y;
    velocity[id].z -= density[id] * correction.z;
}

__global__ static void scaleDensity(int n, double *density, const real factor)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n)
        density[id] *= factor;
}

__global__ static void accumulate(int n, int fieldComponents, const real *src, double *dst)
{
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < n * fieldComponents)
        dst[id] += src[id];
}

} // namespace SamplingHelpersKernels
