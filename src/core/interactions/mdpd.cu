#include "mdpd.h"

#include "pairwise.impl.h"
#include "pairwise_interactions/mdpd.h"

#include <core/celllist.h>
#include <core/utils/common.h>
#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>

#include <memory>


InteractionMDPD::InteractionMDPD(const YmrState *state, std::string name, float rc, float rd, float a, float b, float gamma, float kbt, float power, bool allocateImpl) :
    Interaction(state, name, rc),
    rd(rd), a(a), b(b), gamma(gamma), kbt(kbt), power(power)
{
    if (allocateImpl) {
        PairwiseMDPD mdpd(rc, rd, a, b, gamma, kbt, state->dt, power);
        impl = std::make_unique<InteractionPair<PairwiseMDPD>> (state, name, rc, mdpd);
    }
}

InteractionMDPD::InteractionMDPD(const YmrState *state, std::string name, float rc, float rd, float a, float b, float gamma, float kbt, float power) :
    InteractionMDPD(state, name, rc, rd, a, b, gamma, kbt, power, true)
{}

InteractionMDPD::~InteractionMDPD() = default;

void InteractionMDPD::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2, cl1, cl2);

    pv1->requireDataPerParticle<float>(ChannelNames::densities, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<float>(ChannelNames::densities, DataManager::PersistenceMode::None);
    
    cl1->requireExtraDataPerParticle<float>(ChannelNames::densities);
    cl2->requireExtraDataPerParticle<float>(ChannelNames::densities);
}

std::vector<Interaction::InteractionChannel> InteractionMDPD::getIntermediateInputChannels() const
{
    return {{ChannelNames::densities, Interaction::alwaysActive}};
}

std::vector<Interaction::InteractionChannel> InteractionMDPD::getFinalOutputChannels() const
{
    return impl->getFinalOutputChannels();
}

void InteractionMDPD::local(ParticleVector *pv1, ParticleVector *pv2,
                            CellList *cl1, CellList *cl2,
                            cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

void InteractionMDPD::halo(ParticleVector *pv1, ParticleVector *pv2,
                           CellList *cl1, CellList *cl2,
                           cudaStream_t stream)
{
    impl->halo(pv1, pv2, cl1, cl2, stream);
}

void InteractionMDPD::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                      float a, float b, float gamma, float kbt, float power)
{
    if (a     == Default) a     = this->a;
    if (b     == Default) b     = this->b;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;

    PairwiseMDPD mdpd(this->rc, this->rd, a, b, gamma, kbt, state->dt, power);
    auto ptr = static_cast< InteractionPair<PairwiseMDPD>* >(impl.get());
    
    ptr->setSpecificPair(pv1->name, pv2->name, mdpd);
}


