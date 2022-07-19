// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "dpd.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseDPDInteraction::PairwiseDPDInteraction(const MirState *state,
                                               const std::string& name,
                                               real rc,
                                               DPDParams params)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{}

void PairwiseDPDInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                   CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeLocalInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

void PairwiseDPDInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                                  CellList *cl2, cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeHaloInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

} // namespace mirheo
