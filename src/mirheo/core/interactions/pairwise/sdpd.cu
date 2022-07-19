// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "sdpd.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

template<class PressureEOS, class DensityKernel>
PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
PairwiseSDPDInteraction(const MirState *state,
                        const std::string& name,
                        real rc,
                        SDPDParams params)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{}

template<class PressureEOS, class DensityKernel>
void PairwiseSDPDInteraction<PressureEOS, DensityKernel>::setPrerequisites(ParticleVector *pv1,
                                                                           ParticleVector *pv2,
                                                                           CellList *cl1,
                                                                           CellList *cl2)
{
    pv1->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle<real>(channel_names::densities);
    cl2->requireExtraDataPerParticle<real>(channel_names::densities);
}

template<class PressureEOS, class DensityKernel>
void PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
local(ParticleVector *pv1, ParticleVector *pv2,
      CellList *cl1, CellList *cl2,
      cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeLocalInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

template<class PressureEOS, class DensityKernel>
void PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
halo(ParticleVector *pv1, ParticleVector *pv2,
     CellList *cl1, CellList *cl2,
     cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeHaloInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

template<class PressureEOS, class DensityKernel>
std::vector<Interaction::InteractionChannel> PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
getInputChannels() const
{
    return {{channel_names::densities, Interaction::alwaysActive}};
}



std::unique_ptr<BasePairwiseInteraction>
makePairwiseSDPDInteraction(const MirState *state, const std::string& name, real rc, SDPDParams params)
{
    return std::visit([=](auto eosParams, auto densityKernelParams)
                      -> std::unique_ptr<BasePairwiseInteraction>
    {
        using EOSParamsType = decltype(eosParams);
        using DensityParamsType = decltype(densityKernelParams);
        using PressureEOS = typename EOSParamsType::KernelType;
        using DensityKernel = typename DensityParamsType::KernelType;
        return std::make_unique<PairwiseSDPDInteraction<PressureEOS,DensityKernel>>(state, name, rc, params);
    }, params.varEOSParams, params.varDensityKernelParams);
}


} // namespace mirheo
