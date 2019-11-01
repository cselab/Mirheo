#pragma once

#include "kernels/parameters.h"
#include "drivers_forces.h"
#include "drivers_states.h"

#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/views/rv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

template <int Nstates>
static void updateStatesAndApplyForces(__UNUSED RodVector *rv,
                                       __UNUSED const GPU_RodBiSegmentParameters<Nstates> devParams,
                                       __UNUSED StatesParametersNone& stateParams,
                                       __UNUSED cudaStream_t stream)
{}


template <int Nstates>
static void updateStatesAndApplyForces(RodVector *rv,
                                       const GPU_RodBiSegmentParameters<Nstates> devParams,
                                       StatesSmoothingParameters& stateParams, cudaStream_t stream)
{
    RVview view(rv, rv->local());

    auto kappa = rv->local()->dataPerBisegment.getData<real4>(ChannelNames::rodKappa)->devPtr();
    auto tau_l = rv->local()->dataPerBisegment.getData<real2>(ChannelNames::rodTau_l)->devPtr();

    int nthreads = 128;
    int nblocks = view.nObjects;

    SAFE_KERNEL_LAUNCH(RodStatesKernels::findPolymorphicStates<Nstates>,
                       nblocks, nthreads, 0, stream,
                       view, devParams, kappa, tau_l);

    nthreads = 128;
    nblocks  = getNblocks(view.nObjects * (view.nSegments-1), nthreads);
    
    SAFE_KERNEL_LAUNCH(RodForcesKernels::computeRodCurvatureSmoothing,
                       nblocks, nthreads, 0, stream,
                       view, stateParams.kSmoothing, kappa, tau_l);    
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
static void updateStatesAndApplyForces(RodVector *rv,
                                       const GPU_RodBiSegmentParameters<Nstates> devParams,
                                       StatesSpinParameters& stateParams, cudaStream_t stream)
{
    auto lrv = rv->local();
    RVview view(rv, lrv);

    auto kappa = lrv->dataPerBisegment.getData<real4>(ChannelNames::rodKappa)->devPtr();
    auto tau_l = lrv->dataPerBisegment.getData<real2>(ChannelNames::rodTau_l)->devPtr();

    auto& states = *lrv->dataPerBisegment.getData<int>(ChannelNames::polyStates);
    states.clear(stream);

    // initialize to ground energies without spin interactions
    {
        const int nthreads = 128;
        const int nblocks = view.nObjects;
        
        SAFE_KERNEL_LAUNCH(RodStatesKernels::findPolymorphicStates<Nstates>,
                           nblocks, nthreads, 0, stream,
                           view, devParams, kappa, tau_l);
    }
    
    const int nthreads = 512;
    const int nblocks = view.nObjects;

    // TODO check if it fits into shared mem
    const size_t shMemSize = sizeof(states[0]) * (view.nSegments - 1);
    
    for (int i = 0; i < stateParams.nsteps; ++i)
    {
        auto devSpinParams = getGPUParams(stateParams);
        
        SAFE_KERNEL_LAUNCH(RodStatesKernels::findPolymorphicStatesMCStep<Nstates>,
                           nblocks, nthreads, shMemSize, stream,
                           view, devParams, devSpinParams, kappa, tau_l);
    }
}

} // namespace mirheo
