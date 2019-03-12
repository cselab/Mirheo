#include <memory>

#include "lj_with_stress.h"
#include "pairwise_with_stress.impl.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionLJWithStress::InteractionLJWithStress(const YmrState *state, std::string name,
                                                 float rc, float epsilon, float sigma, float maxForce, bool objectAware, float stressPeriod) :
    InteractionLJ(state, name, rc, epsilon, sigma, maxForce, objectAware, false),
    stressPeriod(stressPeriod)
{
    if (objectAware) {
        PairwiseLJObjectAware lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair_withStress<PairwiseLJObjectAware>> (state, name, rc, stressPeriod, lj);
    }
    else {
        PairwiseLJ lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair_withStress<PairwiseLJ>> (state, name, rc, stressPeriod, lj);
    }
}

InteractionLJWithStress::~InteractionLJWithStress() = default;

void InteractionLJWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                              float epsilon, float sigma, float maxForce)
{
    if (objectAware) {
        PairwiseLJObjectAware lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair_withStress<PairwiseLJObjectAware>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
    else {
        PairwiseLJ lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair_withStress<PairwiseLJ>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
}
