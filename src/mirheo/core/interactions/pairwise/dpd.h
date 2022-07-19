// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_pairwise.h"
#include "kernels/dpd.h"

namespace mirheo {

class PairwiseDPDInteraction: public BasePairwiseInteraction
{
public:
    PairwiseDPDInteraction(const MirState *state, const std::string& name, real rc, DPDParams params);

    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

private:
    PairwiseDPD pair_;
};

} // namespace mirheo
