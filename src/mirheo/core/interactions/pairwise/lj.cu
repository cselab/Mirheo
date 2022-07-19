// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "lj.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseLJInteraction::PairwiseLJInteraction(const MirState *state,
                                             const std::string& name,
                                             real rc,
                                             LJParams params)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{}

void PairwiseLJInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                  CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeLocalInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

void PairwiseLJInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                                 CellList *cl2, cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeHaloInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

} // namespace mirheo
