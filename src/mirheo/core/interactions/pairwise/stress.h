// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/stress_wrapper.h"

#include <mirheo/core/interactions/interface.h>

namespace mirheo {

/** \brief Helper class to control when the stress must be computed.

    In theory stresses could be computed at every time step, but with a higher cost
    (higher bandwidth needed and more computation).
    In practice stress does not need to be computed at every iteration.
    Instead we compute stresses every given periods of simulation time.
 */
class StressManager
{
public:
    /** Construct a StressManager object.
        \param [in] stressPeriod Time (in simulation units) between two stress computations.
     */
    StressManager(real stressPeriod)
        : stressPeriod_(stressPeriod)
    {}

    bool isStressTime(const MirState *state) const
    {
        const auto t = static_cast<real>(state->currentTime);
        return (lastStressTime_+stressPeriod_ <= t) || (lastStressTime_ == t);
    }

    /** Compute local interactions between two ParticleVector. The stresses are computed only when needed.
        \tparam PairwiseKernel The symmetric pairwise kernel of the interaction.
        \param state The global state of the simulation.
        \param [in,out] pair The interaction kernel (without stresses).
        \param [in,out] pairWithStress The interaction kernel (with stresses).
        \param [in,out] pv1 First ParticleVector.
        \param [in,out] pv2 Second ParticleVector.
        \param [in,out] cl1 CellList of pv1.
        \param [in,out] cl2 CellList of pv2.
        \param [in] stream Stream of execution.
     */
    template<class PairwiseKernel>
    void computeLocalInteractions(const MirState *state,
                                  PairwiseKernel& pair,
                                  PairwiseStressWrapper<PairwiseKernel>& pairWithStress,
                                  ParticleVector *pv1, ParticleVector *pv2,
                                  CellList *cl1, CellList *cl2,
                                  cudaStream_t stream);

    /** Computehal halo interactions between two ParticleVector. The stresses are computed only when needed.
        \tparam PairwiseKernel The symmetric pairwise kernel of the interaction.
        \param state The global state of the simulation.
        \param [in,out] pair The interaction kernel (without stresses).
        \param [in,out] pairWithStress The interaction kernel (with stresses).
        \param [in,out] pv1 First ParticleVector.
        \param [in,out] pv2 Second ParticleVector.
        \param [in,out] cl1 CellList of pv1.
        \param [in,out] cl2 CellList of pv2.
        \param [in] stream Stream of execution.
     */
    template<class PairwiseKernel>
    void computeHaloInteractions(const MirState *state,
                                 PairwiseKernel& pair,
                                 PairwiseStressWrapper<PairwiseKernel>& pairWithStress,
                                 ParticleVector *pv1, ParticleVector *pv2,
                                 CellList *cl1, CellList *cl2,
                                 cudaStream_t stream);

    /** \return InteractioChannel for the output of a pairwise Interaction with predicate active when stresses are needed.
        \param state The global state of the simulation.
     */
    Interaction::InteractionChannel getStressPredicate(const MirState *state) const
    {
        auto activePredicateStress = [state,this]()
        {
            return this->isStressTime(state);
        };

        return {channel_names::stresses, activePredicateStress};
    }

private:
    real stressPeriod_;          ///< The stress will be computed every this amount of time
    real lastStressTime_ {-1e6}; ///< to keep track of the last time stress was computed
};


} // namespace mirheo
