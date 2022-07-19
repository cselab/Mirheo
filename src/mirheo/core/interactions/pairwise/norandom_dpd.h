// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/norandom_dpd.h"

namespace mirheo {

class PairwiseNoRandomDPDInteraction: public BasePairwiseInteraction
{
public:
    PairwiseNoRandomDPDInteraction(const MirState *state, const std::string& name, real rc, NoRandomDPDParams params);

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

private:
    PairwiseNoRandomDPD pair_;
};

} // namespace mirheo
