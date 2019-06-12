#include "density.h"

#include "pairwise.impl.h"
#include "pairwise_interactions/density.h"

#include <core/celllist.h>
#include <core/utils/common.h>
#include <core/pvs/particle_vector.h>

#include <memory>


BasicInteractionDensity::BasicInteractionDensity(const YmrState *state, std::string name, float rc) :
    Interaction(state, name, rc)
{}

BasicInteractionDensity::~BasicInteractionDensity() = default;

void BasicInteractionDensity::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2, cl1, cl2);

    pv1->requireDataPerParticle<float>(ChannelNames::densities, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<float>(ChannelNames::densities, DataManager::PersistenceMode::None);
    
    cl1->requireExtraDataPerParticle<float>(ChannelNames::densities);
    cl2->requireExtraDataPerParticle<float>(ChannelNames::densities);
}

std::vector<Interaction::InteractionChannel> BasicInteractionDensity::getIntermediateOutputChannels() const
{
    return {{ChannelNames::densities, Interaction::alwaysActive}};
}
std::vector<Interaction::InteractionChannel> BasicInteractionDensity::getFinalOutputChannels() const
{
    return {};
}

void BasicInteractionDensity::local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

void BasicInteractionDensity::halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    impl->halo(pv1, pv2, cl1, cl2, stream);
}


template <class DensityKernel>
InteractionDensity<DensityKernel>::InteractionDensity(const YmrState *state, std::string name, float rc,
                                                      DensityKernel densityKernel) :
    BasicInteractionDensity(state, name, rc),
    densityKernel(densityKernel)
{
    using PairwiseDensityType = PairwiseDensity<DensityKernel>;
    
    PairwiseDensityType density(rc, DensityKernel());
    impl = std::make_unique<InteractionPair<PairwiseDensityType>> (state, name, rc, density);
}

template <class DensityKernel>
InteractionDensity<DensityKernel>::~InteractionDensity() = default;


template class InteractionDensity<SimpleMDPDDensityKernel>;
template class InteractionDensity<WendlandC2DensityKernel>;
