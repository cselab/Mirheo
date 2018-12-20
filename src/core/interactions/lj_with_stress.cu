#include <memory>

#include "lj_with_stress.h"
#include "pairwise_with_stress.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionLJWithStress::InteractionLJWithStress(std::string name, const YmrState *state, std::string stressName,
                                                 float rc, float epsilon, float sigma, float maxForce, bool objectAware, float stressPeriod) :
    InteractionLJ(name, state, rc, epsilon, sigma, maxForce, objectAware, false),
    stressPeriod(stressPeriod)
{
    if (objectAware) {
        Pairwise_LJObjectAware lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair_withStress<Pairwise_LJObjectAware>> (name, state, stressName, rc, stressPeriod, lj);
    }
    else {
        Pairwise_LJ lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair_withStress<Pairwise_LJ>> (name, state, stressName, rc, stressPeriod, lj);
    }
}

InteractionLJWithStress::~InteractionLJWithStress() = default;

void InteractionLJWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                              float epsilon, float sigma, float maxForce)
{
    if (objectAware) {
        Pairwise_LJObjectAware lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair_withStress<Pairwise_LJObjectAware>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
    else {
        Pairwise_LJ lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair_withStress<Pairwise_LJ>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
}
