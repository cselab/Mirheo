// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/morse.h"
#include "stress.h"

#include <memory>
#include <optional>

namespace mirheo {

/** \brief Interaction to compute forces arising from the Morse potential between particles.
    \tparam Awareness To control which particles interact with which particle
                      (e.g. avoiding interactions between particles of the same object).
 */
template<class Awareness>
class PairwiseMorseInteraction: public BasePairwiseInteraction
{
public:
    /** Create a PairwiseMorseInteraction object.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
        \param [in] params The parameters of the forces.
        \param [in] stressPeriod The simulation time between two stress computations.
                       If set to `std::nullopt`, disables stress computation.
     */
    PairwiseMorseInteraction(const MirState *state, const std::string& name,
                             real rc, MorseParams params,
                             std::optional<real> stressPeriod=std::nullopt);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

    std::vector<InteractionChannel> getOutputChannels() const override;

private:
    PairwiseMorse<Awareness> pair_;
    std::optional<PairwiseStressWrapper<PairwiseMorse<Awareness>>> pairWithStress_;
    std::optional<StressManager> stressManager_;

};


/// Factory helper to instantiate all combinations of Awareness.
std::unique_ptr<BasePairwiseInteraction>
makePairwiseMorseInteraction(const MirState *state, const std::string& name, real rc,
                             MorseParams params, std::optional<real> stressPeriod);

} // namespace mirheo
