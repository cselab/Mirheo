#pragma once

#include "parameters.h"
#include "states_kernels.h"

#include <core/pvs/rod_vector.h>
#include <core/pvs/views/rv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

template <int Nstates>
static void updateStates(RodVector *rv, const GPU_RodBiSegmentParameters<Nstates> devParams,
                         StatesParametersNone& stateParams, cudaStream_t stream)
{}


template <int Nstates>
static void updateStates(RodVector *rv, const GPU_RodBiSegmentParameters<Nstates> devParams,
                         StatesSmoothingParameters& stateParams, cudaStream_t stream)
{
    RVview view(rv, rv->local());

    auto kappa = rv->local()->dataPerBisegment.getData<float4>(ChannelNames::rodKappa)->devPtr();
    auto tau_l = rv->local()->dataPerBisegment.getData<float2>(ChannelNames::rodTau_l)->devPtr();

    const int nthreads = 128;
    const int nblocks = view.nObjects;

    SAFE_KERNEL_LAUNCH(RodStatesKernels::findPolymorphicStates<Nstates>,
                       nblocks, nthreads, 0, stream,
                       view, devParams, kappa, tau_l);
}

static auto getGPUParams(StatesSpinParameters& p)
{
    GPU_SpinParameters dp;
    dp.J    = p.J;
    dp.kBT  = p.kBT;
    dp.beta = 1.0 /  p.kBT;
    dp.seed = p.generate();
    return dp;
}

template <int Nstates>
static void updateStates(RodVector *rv, const GPU_RodBiSegmentParameters<Nstates> devParams,
                         StatesSpinParameters& stateParams, cudaStream_t stream)
{
    RVview view(rv, rv->local());

    auto kappa = rv->local()->dataPerBisegment.getData<float4>(ChannelNames::rodKappa)->devPtr();
    auto tau_l = rv->local()->dataPerBisegment.getData<float2>(ChannelNames::rodTau_l)->devPtr();

    rv->local()->dataPerBisegment.getData<int>(ChannelNames::polyStates)->clear(stream);

    const int nthreads = 512;
    const int nblocks = view.nObjects;

    size_t shared_size = sizeof(int) * (view.nSegments - 1);
    
    for (int i = 0; i < stateParams.nsteps; ++i)
    {
        auto devSpinParams = getGPUParams(stateParams);
        
        SAFE_KERNEL_LAUNCH(RodStatesKernels::findPolymorphicStatesMCStep<Nstates>,
                           nblocks, nthreads, shared_size, stream,
                           view, devParams, devSpinParams, kappa, tau_l);
    }
}


