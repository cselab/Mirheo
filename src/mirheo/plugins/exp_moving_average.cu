// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "exp_moving_average.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/type_add.h>

namespace mirheo {
namespace exp_moving_average_kernels {

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

__D__ RigidMotion operator + (RigidMotion a, RigidMotion b)
{
    a += b;
    return a;
}

__D__ COMandExtent operator + (COMandExtent a, COMandExtent b)
{
    a += b;
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




__D__ RigidMotion operator * (real b, RigidMotion a)
{
    a *= b;
    return a;
}

__D__ COMandExtent operator * (real b, COMandExtent a)
{
    a *= b;
    return a;
}

__D__ Force operator * (real b, Force a)
{
    a *= b;
    return a;
}

__D__ Stress operator * (real b, Stress a)
{
    a *= b;
    return a;
}

template <class T>
__global__ void update(int n, real alpha, const T *src, T *ema)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n)
        ema[i] = alpha * src[i] + (1.0_r - alpha) * ema[i];
}

} // namespace exp_moving_average_kernels

ExponentialMovingAveragePlugin::ExponentialMovingAveragePlugin(const MirState *state, std::string name,
                                                               std::string pvName, real alpha,
                                                               std::string srcChannelName, std::string emaChannelName) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    pv_(nullptr),
    alpha_(alpha),
    srcChannelName_(srcChannelName),
    emaChannelName_(emaChannelName)
{
    channel_names::failIfReserved(emaChannelName_, channel_names::reservedParticleFields);
}

void ExponentialMovingAveragePlugin::beforeIntegration(cudaStream_t stream)
{
    auto& dataManager = pv_->local()->dataPerParticle;
    const auto& srcDesc = dataManager.getChannelDescOrDie(srcChannelName_);
    const auto& emaDesc = dataManager.getChannelDescOrDie(emaChannelName_);

    std::visit([&](auto srcBufferPtr)
    {
        auto *emaBufferPtr = std::get<decltype(srcBufferPtr)>(emaDesc.varDataPtr);

        const int n = srcBufferPtr->size();
        constexpr int nthreads = 128;
        const int nblocks = getNblocks(n, nthreads);

        SAFE_KERNEL_LAUNCH(
            exp_moving_average_kernels::update,
            nblocks, nthreads, 0, stream,
            n, alpha_, srcBufferPtr->devPtr(), emaBufferPtr->devPtr());

    }, srcDesc.varDataPtr);
}

bool ExponentialMovingAveragePlugin::needPostproc()
{
    return false;
}

void ExponentialMovingAveragePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    const auto& desc = pv_->local()->dataPerParticle.getChannelDescOrDie(srcChannelName_);

    std::visit([&](auto pinnedBufferPtr)
    {
        using T = typename std::remove_reference< decltype(*pinnedBufferPtr->hostPtr()) >::type;
        pv_->requireDataPerParticle<T>(emaChannelName_, DataManager::PersistenceMode::Active);
        pv_->local()->dataPerParticle.getData<T>(emaChannelName_)->clearDevice(defaultStream);
    }, desc.varDataPtr);
}

} // namespace mirheo
