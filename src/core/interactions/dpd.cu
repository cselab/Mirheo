#include "dpd.h"
#include <memory>
#include "pairwise.h"
#include "pairwise_interactions/dpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionDPD::InteractionDPD(std::string name, float rc, float a, float gamma, float kbt, float dt, float power) :
    Interaction(name, rc),
    a(a), gamma(gamma), kbt(kbt), dt(dt), power(power)
{
    Pairwise_DPD dpd(rc, a, gamma, kbt, dt, power);
    impl = std::make_unique<InteractionPair<Pairwise_DPD>> (name, rc, dpd);
}

void InteractionDPD::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
    impl->setPrerequisites(pv1, pv2);
}

void InteractionDPD::regular(ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             const float t, cudaStream_t stream)
{
    impl->regular(pv1, pv2, cl1, cl2, t, stream);
}

void InteractionDPD::halo   (ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             const float t, cudaStream_t stream)
{
    impl->halo   (pv1, pv2, cl1, cl2, t, stream);
}

void InteractionDPD::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
        float a, float gamma, float kbt, float dt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (dt    == Default) dt    = this->dt;
    if (power == Default) power = this->power;

    Pairwise_DPD dpd(this->rc, a, gamma, kbt, dt, power);
    auto ptr = static_cast< InteractionPair<Pairwise_DPD>* >(impl.get());
    
    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}

InteractionDPD::~InteractionDPD() = default;

