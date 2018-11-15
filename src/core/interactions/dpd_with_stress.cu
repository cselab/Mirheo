#include <memory>

#include "dpd_with_stress.h"
#include "pairwise_with_stress.h"
#include "pairwise_interactions/dpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionDPDWithStress::InteractionDPDWithStress(std::string name, float rc, float a, float gamma, float kbt, float dt, float power, float stressPeriod) :
    InteractionDPD(name, rc, a, gamma, kbt, dt, power, false),
    stressPeriod(stressPeriod)
{
    Pairwise_DPD dpd(rc, a, gamma, kbt, dt, power);
    impl = std::make_unique<InteractionPair_withStress<Pairwise_DPD>> (name, rc, stressPeriod, dpd);
}

InteractionDPDWithStress::~InteractionDPDWithStress() = default;

void InteractionDPDWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                               float a, float gamma, float kbt, float dt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (dt    == Default) dt    = this->dt;
    if (power == Default) power = this->power;

    Pairwise_DPD dpd(this->rc, a, gamma, kbt, dt, power);
    auto ptr = static_cast< InteractionPair_withStress<Pairwise_DPD>* >(impl.get());
    
    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}
