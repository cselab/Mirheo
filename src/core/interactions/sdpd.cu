#include "sdpd.h"

#include "pairwise.impl.h"
#include "pairwise_interactions/density.h"
#include "pairwise_interactions/sdpd.h"

#include <core/celllist.h>
#include <core/utils/common.h>
#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>

#include <memory>

BasicInteractionSDPD::BasicInteractionSDPD(const YmrState *state, std::string name, float rc,
                                           float viscosity, float kBT) :
    Interaction(state, name, rc),
    viscosity(viscosity),
    kBT(kBT)
{}

BasicInteractionSDPD::~BasicInteractionSDPD() = default;

void BasicInteractionSDPD::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2, cl1, cl2);

    pv1->requireDataPerParticle<float>(ChannelNames::densities, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<float>(ChannelNames::densities, DataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle<float>(ChannelNames::densities);
    cl2->requireExtraDataPerParticle<float>(ChannelNames::densities);
}

std::vector<Interaction::InteractionChannel> BasicInteractionSDPD::getIntermediateInputChannels() const
{
    return {{ChannelNames::densities, Interaction::alwaysActive}};
}

std::vector<Interaction::InteractionChannel> BasicInteractionSDPD::getFinalOutputChannels() const
{
    return impl->getFinalOutputChannels();
}

void BasicInteractionSDPD::local(ParticleVector *pv1, ParticleVector *pv2,
                            CellList *cl1, CellList *cl2,
                            cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

void BasicInteractionSDPD::halo(ParticleVector *pv1, ParticleVector *pv2,
                           CellList *cl1, CellList *cl2,
                           cudaStream_t stream)
{
    impl->halo(pv1, pv2, cl1, cl2, stream);
}





template <class PressureEOS, class DensityKernel>
InteractionSDPD<PressureEOS, DensityKernel>::InteractionSDPD(const YmrState *state, std::string name, float rc,
                                                             PressureEOS pressure, DensityKernel densityKernel,
                                                             float viscosity, float kBT, bool allocateImpl) :
    BasicInteractionSDPD(state, name, rc, viscosity, kBT),
    pressure(pressure),
    densityKernel(densityKernel)
{
    if (allocateImpl) {
        using pairwiseType = PairwiseSDPD<PressureEOS, DensityKernel>;
        pairwiseType sdpd(rc, pressure, densityKernel, viscosity, kBT, state->dt);
        impl = std::make_unique<InteractionPair<pairwiseType>> (state, name, rc, sdpd);
    }
}

template <class PressureEOS, class DensityKernel>
InteractionSDPD<PressureEOS, DensityKernel>::InteractionSDPD(const YmrState *state, std::string name, float rc,
                                                             PressureEOS pressure, DensityKernel densityKernel,
                                                             float viscosity, float kBT) :
    InteractionSDPD(state, name, rc, pressure, densityKernel, viscosity, kBT, true)
{}

template <class PressureEOS, class DensityKernel>
InteractionSDPD<PressureEOS, DensityKernel>::~InteractionSDPD() = default;


template class InteractionSDPD<LinearPressureEOS, WendlandC2DensityKernel>;
template class InteractionSDPD<QuasiIncompressiblePressureEOS, WendlandC2DensityKernel>;
