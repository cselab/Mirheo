// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "norandom_dpd.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseNoRandomDPDInteraction::PairwiseNoRandomDPDInteraction(const MirState *state,
                                                               const std::string& name,
                                                               real rc,
                                                               NoRandomDPDParams params)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{}

void PairwiseNoRandomDPDInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                           CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    symmetric_pairwise_helpers::computeLocalInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
}

void PairwiseNoRandomDPDInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                                          CellList *cl2, cudaStream_t stream)
{
    symmetric_pairwise_helpers::computeHaloInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
}

} // namespace mirheo
