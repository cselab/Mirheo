// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "density.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

template<class Awareness>
PairwiseDensityInteraction<Awareness>::PairwiseDensityInteraction(const MirState *state,
                                                                  const std::string& name,
                                                                  real rc,
                                                                  DensityParams params)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{}

template<class Awareness>
void PairwiseDensityInteraction<Awareness>::setPrerequisites(ParticleVector *pv1,
                                                             ParticleVector *pv2,
                                                             CellList *cl1,
                                                             CellList *cl2)
{
    pv1->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle<real>(channel_names::densities);
    cl2->requireExtraDataPerParticle<real>(channel_names::densities);
}

template<class Awareness>
void PairwiseDensityInteraction<Awareness>::local(ParticleVector *pv1, ParticleVector *pv2,
                                                  CellList *cl1, CellList *cl2,
                                                  cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeLocalInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

template<class Awareness>
void PairwiseDensityInteraction<Awareness>::halo(ParticleVector *pv1, ParticleVector *pv2,
                                                 CellList *cl1, CellList *cl2,
                                                 cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeHaloInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}


template<class Awareness>
Interaction::Stage PairwiseDensityInteraction<Awareness>::getStage() const
{
    return Stage::Intermediate;
}

template<class Awareness>
std::vector<Interaction::InteractionChannel>
PairwiseDensityInteraction<Awareness>::getOutputChannels() const
{
    return {{channel_names::densities, Interaction::alwaysActive}};
}


std::unique_ptr<BasePairwiseInteraction>
makePairwiseDensityInteraction(const MirState *state, const std::string& name, real rc, DensityParams params)
{
    return std::visit([=](auto densityParams) -> std::unique_ptr<BasePairwiseInteraction>
    {
        using AwarenessParamsType = decltype(densityParams);
        using Awareness = typename AwarenessParamsType::KernelType;
        return std::make_unique<PairwiseDensityInteraction<Awareness>>(state, name, rc, params);
    }, params.varDensityKernelParams);
}


} // namespace mirheo
