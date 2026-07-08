// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "particle_channel_averager.h"

#include "utils/time_stamp.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/type_add.h>

namespace mirheo
{

namespace particle_channel_averager_kernels
{

/// do nothing, it does not make sense.
__D__ RigidMotion& operator += (RigidMotion& a, RigidMotion __UNUSED b)
{
    return a;
}

__D__ COMandExtent& operator += (COMandExtent& a, COMandExtent b)
{
    a.com += b.com;
    a.low += b.low;
    a.high += b.high;
    return a;
}


/// do nothing, it does not make sense.
__D__ RigidMotion& operator *= (RigidMotion& a, real __UNUSED s)
{
    return a;
}

__D__ COMandExtent& operator *= (COMandExtent& a, real b)
{
    a.com *= b;
    a.low *= b;
    a.high *= b;
    return a;
}

__D__ Force& operator *= (Force& a, real b)
{
    a.f *= b;
    return a;
}

__D__ Stress& operator *= (Stress& a, real b)
{
    a.xx *= b;
    a.xy *= b;
    a.xz *= b;
    a.yy *= b;
    a.yz *= b;
    a.zz *= b;
    return a;
}

template <class T>
__global__ void add(int n, const T *src, T *sum)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n)
        sum[i] += src[i];
}

template <class T>
__global__ void computeAverage(int n, const T *sum, real scale, T *average)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n)
    {
        auto s = sum[i];
        s *= scale;
        average[i] = s;
    }
}

} // namespace particle_channel_averager_kernels

ParticleChannelAveragerPlugin::ParticleChannelAveragerPlugin(const MirState *state, std::string name, std::string pvName,
                                                             std::string channelName, std::string averageName, real updateEvery):

    SimulationPlugin(state, name),
    pvName_(pvName),
    channelName_(channelName),
    averageName_(averageName),
    updateEvery_(updateEvery)
{
    channel_names::failIfReserved(averageName_, channel_names::reservedParticleFields);
    sumName_ = averageName_ + "Sum__";
}

void ParticleChannelAveragerPlugin::beforeIntegration(cudaStream_t stream)
{
    auto& dataManager = pv_->local()->dataPerParticle;
    const auto& srcDesc = dataManager.getChannelDescOrDie(channelName_);
    const auto& sumDesc = dataManager.getChannelDescOrDie(sumName_);

    std::visit([&](auto srcBufferPtr)
    {
        auto *sumBufferPtr = std::get<decltype(srcBufferPtr)>(sumDesc.varDataPtr);

        const int n = srcBufferPtr->size();
        constexpr int nthreads = 128;
        const int nblocks = getNblocks(n, nthreads);

        SAFE_KERNEL_LAUNCH(
            particle_channel_averager_kernels::add,
            nblocks, nthreads, 0, stream,
            n, srcBufferPtr->devPtr(), sumBufferPtr->devPtr());

    }, srcDesc.varDataPtr);


    ++ nSamples_;

    if (isTimeEvery(getState(), updateEvery_))
    {
        // Compute averages and reinitialize the sum
        const auto& averageDesc = dataManager.getChannelDescOrDie(averageName_);
        const real scale = 1.0_r / nSamples_;

        std::visit([&](auto sumBufferPtr)
        {
            auto *averageBufferPtr = std::get<decltype(sumBufferPtr)>(averageDesc.varDataPtr);

            const int n = sumBufferPtr->size();
            constexpr int nthreads = 128;
            const int nblocks = getNblocks(n, nthreads);

            SAFE_KERNEL_LAUNCH(
                 particle_channel_averager_kernels::computeAverage,
                 nblocks, nthreads, 0, stream,
                 n, sumBufferPtr->devPtr(), scale, averageBufferPtr->devPtr());

            sumBufferPtr->clearDevice(stream);
        }, sumDesc.varDataPtr);

        nSamples_ = 0;
    }
}

bool ParticleChannelAveragerPlugin::needPostproc()
{
    return false;
}

void ParticleChannelAveragerPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    const auto& desc = pv_->local()->dataPerParticle.getChannelDescOrDie(channelName_);

    std::visit([&](auto pinnedBufferPtr)
    {
        using T = typename std::remove_reference< decltype(*pinnedBufferPtr->hostPtr()) >::type;
        pv_->requireDataPerParticle<T>(averageName_, DataManager::PersistenceMode::Active);
        pv_->requireDataPerParticle<T>(sumName_, DataManager::PersistenceMode::Active);
    }, desc.varDataPtr);
}

} // namespace mirheo
