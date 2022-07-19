// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/morse.h"

#include <memory>

namespace mirheo {

template<class Awareness>
class PairwiseMorseInteraction: public BasePairwiseInteraction
{
public:
    PairwiseMorseInteraction(const MirState *state, const std::string& name, real rc, MorseParams params);

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

private:
    PairwiseMorse<Awareness> pair_;
};


/// Factory helper to instantiate all combinations of Awareness.
std::unique_ptr<BasePairwiseInteraction>
makePairwiseMorseInteraction(const MirState *state, const std::string& name, real rc, MorseParams params);

} // namespace mirheo
