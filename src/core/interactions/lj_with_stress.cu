#include "lj_with_stress.h"
#include "pairwise/kernels/lj.h"
#include "pairwise/impl.stress.h"

#include <core/pvs/particle_vector.h>

#include <memory>

InteractionLJWithStress::InteractionLJWithStress(const MirState *state, std::string name,
                                                 float rc, float epsilon, float sigma, float maxForce,
                                                 AwareMode awareness, int minSegmentsDist, float stressPeriod) :
    InteractionLJ(state, name, rc, epsilon, sigma, maxForce, awareness, minSegmentsDist, false),
    stressPeriod(stressPeriod)
{
    if (awareness == AwareMode::None)
    {
        PairwiseLJ<LJAwarenessNone> lj(rc, epsilon, sigma, maxForce, {});
        impl = std::make_unique<InteractionPair_withStress<PairwiseLJ<LJAwarenessNone>>> (state, name, rc, stressPeriod, lj);
    }
    else if (awareness == AwareMode::Object)
    {
        PairwiseLJ<LJAwarenessObject> lj(rc, epsilon, sigma, maxForce, {});
        impl = std::make_unique<InteractionPair_withStress<PairwiseLJ<LJAwarenessObject>>> (state, name, rc, stressPeriod, lj);
    }
    else
    {
        PairwiseLJ<LJAwarenessRod> lj(rc, epsilon, sigma, maxForce, {minSegmentsDist});
        impl = std::make_unique<InteractionPair_withStress<PairwiseLJ<LJAwarenessRod>>> (state, name, rc, stressPeriod, lj);
    }
}

InteractionLJWithStress::~InteractionLJWithStress() = default;

void InteractionLJWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                              float epsilon, float sigma, float maxForce)
{

    if (awareness == AwareMode::None)
    {
        PairwiseLJ<LJAwarenessNone> lj(rc, epsilon, sigma, maxForce, {});
        auto ptr = static_cast< InteractionPair_withStress<PairwiseLJ<LJAwarenessNone>>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
    else if (awareness == AwareMode::Object)
    {
        PairwiseLJ<LJAwarenessObject> lj(rc, epsilon, sigma, maxForce, {});
        auto ptr = static_cast< InteractionPair_withStress<PairwiseLJ<LJAwarenessObject>>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
    else
    {
        PairwiseLJ<LJAwarenessRod> lj(rc, epsilon, sigma, maxForce, {minSegmentsDist});
        auto ptr = static_cast< InteractionPair_withStress<PairwiseLJ<LJAwarenessRod>>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
}
