// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/stress_wrapper.h"

#include <mirheo/core/interactions/interface.h>

namespace mirheo {

class StressManager
{
public:
    StressManager(real stressPeriod)
        : stressPeriod_(stressPeriod)
    {}

    template<class PairwiseKernel>
    void computeLocalInteractions(const MirState *state,
                                  PairwiseKernel& pair,
                                  PairwiseStressWrapper<PairwiseKernel>& pairWithStress,
                                  ParticleVector *pv1, ParticleVector *pv2,
                                  CellList *cl1, CellList *cl2,
                                  cudaStream_t stream);

    template<class PairwiseKernel>
    void computeHaloInteractions(const MirState *state,
                                 PairwiseKernel& pair,
                                 PairwiseStressWrapper<PairwiseKernel>& pairWithStress,
                                 ParticleVector *pv1, ParticleVector *pv2,
                                 CellList *cl1, CellList *cl2,
                                 cudaStream_t stream);

    Interaction::InteractionChannel getStressPredicate(const MirState *state) const
    {
        auto activePredicateStress = [state,this]()
        {
            const real t = state->currentTime;
            return (lastStressTime_+stressPeriod_ <= t) || (lastStressTime_ == t);
        };

        return {channel_names::stresses, activePredicateStress};
    }

private:
    real stressPeriod_;          ///< The stress will be computed every this amount of time
    real lastStressTime_ {-1e6}; ///< to keep track of the last time stress was computed
};


} // namespace mirheo
