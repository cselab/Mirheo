#include "rod.h"
#include "rod.impl.h"


InteractionRod::InteractionRod(const YmrState *state, std::string name, RodParameters parameters) :
    Interaction(state, name, /*rc*/ 1.f)
{
    impl = std::make_unique<InteractionRodImpl>(state, name, parameters);
}

InteractionRod::~InteractionRod() = default;

void InteractionRod::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    if (pv1 != pv2)
        die("Internal rod forces can't be computed between two different particle vectors");

    auto rv = dynamic_cast<RodVector*>(pv1);
    if (rv == nullptr)
        die("Internal rod forces can only be computed with a RodVector");

    impl->setPrerequisites(pv1, pv2, cl1, cl2);
}

void InteractionRod::local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    if (impl.get() == nullptr)
        die("%s needs a concrete implementation, none was provided", name.c_str());

    impl->local(pv1, pv2, cl1, cl2, stream);
}

void InteractionRod::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    debug("Not computing internal rod forces between local and halo rods of '%s'", pv1->name.c_str());
}

bool InteractionRod::isSelfObjectInteraction() const
{
    return true;
}
