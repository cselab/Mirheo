#include "lj.h"
#include "pairwise.impl.h"
#include "pairwise/kernels/lj.h"
#include "pairwise/kernels/lj_object_aware.h"
#include "pairwise/kernels/lj_rod_aware.h"

#include <core/celllist.h>

#include <memory>

InteractionLJ::InteractionLJ(const MirState *state, std::string name, float rc, float epsilon, float sigma, float maxForce,
                             AwareMode awareness, int minSegmentsDist, bool allocate) :
    Interaction(state, name, rc),
    awareness(awareness),
    minSegmentsDist(minSegmentsDist)
{
    if (!allocate) return;

    if (awareness == AwareMode::None)
    {
        PairwiseLJ lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair<PairwiseLJ>> (state, name, rc, lj);
    }
    else if (awareness == AwareMode::Object)
    {
        PairwiseLJObjectAware lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair<PairwiseLJObjectAware>> (state, name, rc, lj);
    }
    else 
    {
        PairwiseLJRodAware lj(rc, epsilon, sigma, maxForce, minSegmentsDist);
        impl = std::make_unique<InteractionPair<PairwiseLJRodAware>> (state, name, rc, lj);
    }
}

InteractionLJ::InteractionLJ(const MirState *state, std::string name, float rc, float epsilon, float sigma, float maxForce,
                             AwareMode awareness, int minSegmentsDist) :
    InteractionLJ(state, name, rc, epsilon, sigma, maxForce, awareness, minSegmentsDist, true)
{}

InteractionLJ::~InteractionLJ() = default;

void InteractionLJ::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2, cl1, cl2);
}

std::vector<Interaction::InteractionChannel> InteractionLJ::getOutputChannels() const
{
    return impl->getOutputChannels();
}

void InteractionLJ::local(ParticleVector *pv1, ParticleVector *pv2,
                          CellList *cl1, CellList *cl2,
                          cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

void InteractionLJ::halo(ParticleVector *pv1, ParticleVector *pv2,
                         CellList *cl1, CellList *cl2,
                         cudaStream_t stream)
{
    impl->halo(pv1, pv2, cl1, cl2, stream);
}

void InteractionLJ::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                    float epsilon, float sigma, float maxForce)
{
    if (awareness == AwareMode::None)
    {
        PairwiseLJ lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair<PairwiseLJ>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
    else if (awareness == AwareMode::Object)
    {
        PairwiseLJObjectAware lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair<PairwiseLJObjectAware>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
    else
    {
        PairwiseLJRodAware lj(rc, epsilon, sigma, maxForce, minSegmentsDist);
        auto ptr = static_cast< InteractionPair<PairwiseLJRodAware>* >(impl.get());
        ptr->setSpecificPair(pv1->name, pv2->name, lj);
    }
}

