// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/repulsive_lj.h"

#include <memory>

namespace mirheo {

template<class Awareness>
class PairwiseRepulsiveLJInteraction: public BasePairwiseInteraction
{
public:
    PairwiseRepulsiveLJInteraction(const MirState *state, const std::string& name, real rc, RepulsiveLJParams params);

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

private:
    PairwiseRepulsiveLJ<Awareness> pair_;
};


/// Factory helper to instantiate all combinations of Awareness.
std::unique_ptr<BasePairwiseInteraction>
makePairwiseRepulsiveLJInteraction(const MirState *state, const std::string& name, real rc, RepulsiveLJParams params);

} // namespace mirheo
