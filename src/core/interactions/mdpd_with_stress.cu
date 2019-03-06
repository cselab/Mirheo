#include <memory>

#include "mdpd_with_stress.h"
#include "pairwise_with_stress.impl.h"
#include "pairwise_interactions/mdpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionMDPDWithStress::InteractionMDPDWithStress(const YmrState *state, std::string name,
                                                     float rc, float rd, float a, float b, float gamma, float kbt, float power,
                                                     float stressPeriod) :
    InteractionMDPD(state, name, rc, rd, a, b, gamma, kbt, power, false),
    stressPeriod(stressPeriod)
{
    Pairwise_MDPD mdpd(rc, rd, a, b, gamma, kbt, state->dt, power);
    impl = std::make_unique<InteractionPair_withStress<Pairwise_MDPD>> (state, name, rc, stressPeriod, mdpd);
}

InteractionMDPDWithStress::~InteractionMDPDWithStress() = default;

void InteractionMDPDWithStress::setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, 
                                                float a, float b, float gamma, float kbt, float power)
{
    if (a     == Default) a     = this->a;
    if (b     == Default) b     = this->b;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;

    Pairwise_MDPD mdpd(this->rc, this->rd, a, b, gamma, kbt, state->dt, power);
    auto ptr = static_cast< InteractionPair_withStress<Pairwise_MDPD>* >(impl.get());
    
    ptr->setSpecificPair(pv1->name, pv2->name, mdpd);
}
