#include "sdpd.h"

#include "pairwise.impl.h"
#include "pairwise_interactions/density.h"
#include "pairwise_interactions/sdpd.h"

#include <core/celllist.h>
#include <core/utils/common.h>
#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>

#include <memory>


template <class DensityKernel>
InteractionSDPDDensity<DensityKernel>::InteractionSDPDDensity(const YmrState *state, std::string name, float rc, DensityKernel densityKernel) :
    Interaction(state, name, rc)
{
    using PairwiseDensityType = PairwiseDensity<DensityKernel>;
    
    PairwiseDensityType density(rc, densityKernel);
    impl = std::make_unique<InteractionPair<PairwiseDensityType>> (state, name, rc, density);
}

template<class DensityKernel>
InteractionSDPDDensity<DensityKernel>::~InteractionSDPDDensity() = default;

template<class DensityKernel>
void InteractionSDPDDensity<DensityKernel>::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2, cl1, cl2);

    pv1->requireDataPerParticle<float>(ChannelNames::densities, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<float>(ChannelNames::densities, ExtraDataManager::PersistenceMode::None);
    
    cl1->requireExtraDataPerParticle<float>(ChannelNames::densities);
    cl2->requireExtraDataPerParticle<float>(ChannelNames::densities);
}

template<class DensityKernel>
std::vector<Interaction::InteractionChannel>
InteractionSDPDDensity<DensityKernel>::getIntermediateOutputChannels() const
{
    return {{ChannelNames::densities, Interaction::alwaysActive}};
}

template<class DensityKernel>
std::vector<Interaction::InteractionChannel>
InteractionSDPDDensity<DensityKernel>::getFinalOutputChannels() const
{
    return {};
}

template<class DensityKernel>
void InteractionSDPDDensity<DensityKernel>::local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

template<class DensityKernel>
void InteractionSDPDDensity<DensityKernel>::halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    impl->halo(pv1, pv2, cl1, cl2, stream);
}





template <class PressureEOS, class DensityKernel>
InteractionSDPD<PressureEOS, DensityKernel>::InteractionSDPD(const YmrState *state, std::string name, float rc,
                                                             PressureEOS pressure, DensityKernel densityKernel,
                                                             float viscosity, float kBT, bool allocateImpl) :
    Interaction(state, name, rc),
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

template <class PressureEOS, class DensityKernel>
void InteractionSDPD<PressureEOS, DensityKernel>::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2, cl1, cl2);

    pv1->requireDataPerParticle<float>(ChannelNames::densities, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<float>(ChannelNames::densities, ExtraDataManager::PersistenceMode::None);
    
    cl1->requireExtraDataPerParticle<float>(ChannelNames::densities);
    cl2->requireExtraDataPerParticle<float>(ChannelNames::densities);
}

template <class PressureEOS, class DensityKernel>
std::vector<Interaction::InteractionChannel>
InteractionSDPD<PressureEOS, DensityKernel>::getIntermediateInputChannels() const
{
    return {{ChannelNames::densities, Interaction::alwaysActive}};
}

template <class PressureEOS, class DensityKernel>
std::vector<Interaction::InteractionChannel>
InteractionSDPD<PressureEOS, DensityKernel>::getFinalOutputChannels() const
{
    return impl->getFinalOutputChannels();
}

template <class PressureEOS, class DensityKernel>
void
InteractionSDPD<PressureEOS, DensityKernel>::local(ParticleVector *pv1, ParticleVector *pv2,
                                                   CellList *cl1, CellList *cl2,
                                                   cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

template <class PressureEOS, class DensityKernel>
void
InteractionSDPD<PressureEOS, DensityKernel>::halo(ParticleVector *pv1, ParticleVector *pv2,
                                                  CellList *cl1, CellList *cl2,
                                                  cudaStream_t stream)
{
    impl->halo(pv1, pv2, cl1, cl2, stream);
}


template class InteractionSDPD<LinearPressureEOS, WendlandC2DensityKernel>;
template class InteractionSDPD<QuasiIncompressiblePressureEOS, WendlandC2DensityKernel>;
