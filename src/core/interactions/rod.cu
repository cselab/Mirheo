#include "rod.h"
#include "rod.impl.h"


InteractionRod::InteractionRod(const YmrState *state, std::string name, RodParameters parameters) :
    Interaction(state, name, /*rc*/ 1.f)
{
    impl = std::make_unique<InteractionRodImpl>(state, name, parameters);
}

InteractionRod::~InteractionRod() = default;
