// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "repulsive_lj.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

template<class Awareness>
PairwiseRepulsiveLJInteraction<Awareness>::PairwiseRepulsiveLJInteraction(const MirState *state,
                                                                          const std::string& name,
                                                                          real rc,
                                                                          RepulsiveLJParams params)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{}

template<class Awareness>
void PairwiseRepulsiveLJInteraction<Awareness>::local(ParticleVector *pv1, ParticleVector *pv2,
                                                      CellList *cl1, CellList *cl2,
                                                      cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeLocalInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

template<class Awareness>
void PairwiseRepulsiveLJInteraction<Awareness>::halo(ParticleVector *pv1, ParticleVector *pv2,
                                                     CellList *cl1, CellList *cl2,
                                                     cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeHaloInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}


std::unique_ptr<BasePairwiseInteraction>
makePairwiseRepulsiveLJInteraction(const MirState *state, const std::string& name, real rc, RepulsiveLJParams params)
{
    return std::visit([=](auto awarenessParams) -> std::unique_ptr<BasePairwiseInteraction>
    {
        using AwarenessParamsType = decltype(awarenessParams);
        using Awareness = typename AwarenessParamsType::KernelType;
        return std::make_unique<PairwiseRepulsiveLJInteraction<Awareness>>(state, name, rc, params);
    }, params.varAwarenessParams);
}


} // namespace mirheo
