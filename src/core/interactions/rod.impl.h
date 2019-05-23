#pragma once

#include "interface.h"
#include "rod/forces_kernels.h"
#include "rod/parameters.h"

#include <core/pvs/rod_vector.h>
#include <core/pvs/views/rv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

static auto getBoundParams(const RodParameters& p)
{
    GPU_RodBoundsParameters dp;
    dp.kBounds = p.kBounds;
    dp.kVisc   = p.kVisc;
    dp.lcenter = p.l0;
    dp.lcross  = p.a0;
    dp.lring   = 0.5 * sqrt(2.0) * p.a0;
    dp.ldiag   = 0.5 * sqrt(p.a0*p.a0 + p.l0*p.l0);
    return dp;
}

template <int Nstates>
static auto getBiSegmentParams(const RodParameters& p)
{
    GPU_RodBiSegmentParameters<Nstates> dp;
    dp.kBending = p.kBending;
    dp.kTwist   = p.kTwist;

    for (size_t i = 0; i < p.kappaEq.size(); ++i)
    {
        dp.kappaEq[i]  = p.kappaEq[i];
        dp.tauEq[i]    = p.tauEq[i];
        dp.groundE[i]  = p.groundE[i];
    }
    return dp;
}

template <int Nstates>
class InteractionRodImpl : public Interaction
{
public:
    InteractionRodImpl(const YmrState *state, std::string name, RodParameters parameters) :
        Interaction(state, name, /* rc */ 1.0f),
        parameters(parameters)
    {}

    ~InteractionRodImpl() = default;
    
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        auto rv = dynamic_cast<RodVector *>(pv1);

        debug("Computing internal rod forces for %d rods of '%s'",
              rv->local()->nObjects, rv->name.c_str());

        rv->updateBishopFrame(stream);
        
        RVview view(rv, rv->local());

        {
            const int nthreads = 128;
            const int nblocks  = getNblocks(view.nObjects * view.nSegments, nthreads);
        
            auto devParams = getBoundParams(parameters);
        
            SAFE_KERNEL_LAUNCH(RodForcesKernels::computeRodBoundForces,
                               nblocks, nthreads, 0, stream,
                               view, devParams);
        }

        {
            const int nthreads = 128;
            const int nblocks  = getNblocks(view.nObjects * (view.nSegments-1), nthreads);
        
            auto devParams = getBiSegmentParams<Nstates>(parameters);
        
            SAFE_KERNEL_LAUNCH(RodForcesKernels::computeRodBiSegmentForces,
                               nblocks, nthreads, 0, stream,
                               view, devParams);
        }
    }

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
    {}
    
protected:

    RodParameters parameters;
};
