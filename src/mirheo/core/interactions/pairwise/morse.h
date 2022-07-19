// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/morse.h"
#include "stress.h"

#include <memory>
#include <optional>

namespace mirheo {

template<class Awareness>
class PairwiseMorseInteraction: public BasePairwiseInteraction
{
public:
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
