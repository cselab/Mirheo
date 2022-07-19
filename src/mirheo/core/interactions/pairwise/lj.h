// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/lj.h"
#include "stress.h"

#include <optional>

namespace mirheo {

class PairwiseLJInteraction: public BasePairwiseInteraction
{
public:
    PairwiseLJInteraction(const MirState *state, const std::string& name,
                          real rc, LJParams params,
                          std::optional<real> stressPeriod=std::nullopt);

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

    std::vector<InteractionChannel> getOutputChannels() const override;

private:
    PairwiseLJ pair_;
    std::optional<PairwiseStressWrapper<PairwiseLJ>> pairWithStress_;
    std::optional<StressManager> stressManager_;
};

} // namespace mirheo
