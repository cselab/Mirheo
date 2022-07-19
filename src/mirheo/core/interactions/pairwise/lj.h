// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/lj.h"

namespace mirheo {

class PairwiseLJInteraction: public BasePairwiseInteraction
{
public:
    PairwiseLJInteraction(const MirState *state, const std::string& name, real rc, LJParams params);

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

private:
    PairwiseLJ pair_;
};

} // namespace mirheo
