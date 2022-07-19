// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "sdpd.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

template<class PressureEOS, class DensityKernel>
PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
PairwiseSDPDInteraction(const MirState *state,
                        const std::string& name,
                        real rc,
                        SDPDParams params,
                        std::optional<real> stressPeriod)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)
{
    if (stressPeriod)
    {
        pairWithStress_ = PairwiseStressWrapper<PairwiseSDPD<PressureEOS,DensityKernel>>(rc, params);
        stressManager_ = StressManager(*stressPeriod);
    }
}

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

    if (stressManager_)
    {
        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
        cl2->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }
}

template<class PressureEOS, class DensityKernel>
void PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
local(ParticleVector *pv1, ParticleVector *pv2,
      CellList *cl1, CellList *cl2,
      cudaStream_t stream)
{
    if (stressManager_)
    {
        stressManager_->computeLocalInteractions(getState(),
                                                 pair_, *pairWithStress_,
                                                 pv1, pv2, cl1, cl2, stream);
    }
    else
    {
        symmetric_pairwise_helpers::computeLocalInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
    }

}

template<class PressureEOS, class DensityKernel>
void PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
halo(ParticleVector *pv1, ParticleVector *pv2,
     CellList *cl1, CellList *cl2,
     cudaStream_t stream)
{
    if (stressManager_)
    {
        stressManager_->computeHaloInteractions(getState(),
                                                pair_, *pairWithStress_,
                                                pv1, pv2, cl1, cl2, stream);
    }
    else
    {
        symmetric_pairwise_helpers::computeHaloInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
    }

}

template<class PressureEOS, class DensityKernel>
std::vector<Interaction::InteractionChannel> PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
getInputChannels() const
{
    return {{channel_names::densities, Interaction::alwaysActive}};
}



template<class PressureEOS, class DensityKernel>
std::vector<Interaction::InteractionChannel> PairwiseSDPDInteraction<PressureEOS, DensityKernel>::
getOutputChannels() const
{
    std::vector<InteractionChannel> channels = {{channel_names::forces, alwaysActive}};

    if (stressManager_)
    {
        channels.push_back(stressManager_->getStressPredicate(getState()));
    }

    return channels;
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
