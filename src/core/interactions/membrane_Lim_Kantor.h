#pragma once

#include "membrane.h"
#include "membrane/parameters.h"

#include <memory>

class InteractionMembraneLimKantor : public InteractionMembrane
{
public:
    InteractionMembraneLimKantor(const YmrState *state, std::string name,
                                 CommonMembraneParameters parameters,
                                 LimParameters limParams,
                                 KantorBendingParameters kantorParams,
                                 bool stressFree, float growUntil);
    
    ~InteractionMembraneLimKantor();
};
