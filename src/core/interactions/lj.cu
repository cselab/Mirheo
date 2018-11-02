#include "lj.h"
#include <memory>
#include "pairwise.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"


InteractionLJ::InteractionLJ(std::string name, float rc,
                             float epsilon, float sigma, float maxForce, bool objectAware) :
    Interaction(name, rc), objectAware(objectAware)
{
    if (objectAware)
    {
        Pairwise_LJObjectAware lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair<Pairwise_LJObjectAware>> (name, rc, lj);
    }
    else
    {
        Pairwise_LJ lj(rc, epsilon, sigma, maxForce);
        impl = std::make_unique<InteractionPair<Pairwise_LJ>> (name, rc, lj);
    }
}

void InteractionLJ::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
    impl->setPrerequisites(pv1, pv2);
}

void InteractionLJ::regular(ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             const float t, cudaStream_t stream)
{
    impl->regular(pv1, pv2, cl1, cl2, t, stream);
}

void InteractionLJ::halo   (ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             const float t, cudaStream_t stream)
{
    impl->halo   (pv1, pv2, cl1, cl2, t, stream);
}

void InteractionLJ::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
        float epsilon, float sigma, float maxForce)
{
    if (objectAware)
    {
        Pairwise_LJObjectAware lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair<Pairwise_LJObjectAware>* >(impl.get());
        ptr->setSpecificPair(pv1->name(), pv2->name(), lj);
    }
    else
    {
        Pairwise_LJ lj(rc, epsilon, sigma, maxForce);
        auto ptr = static_cast< InteractionPair<Pairwise_LJ>* >(impl.get());
        ptr->setSpecificPair(pv1->name(), pv2->name(), lj);
    }
}

InteractionLJ::~InteractionLJ() = default;

