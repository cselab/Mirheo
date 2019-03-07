#pragma once

#include "membrane.h"
#include "membrane/parameters.h"

#include <memory>

class InteractionMembraneWLCKantor : public InteractionMembrane
{
public:
    InteractionMembraneWLCKantor(const YmrState *state, std::string name,
                                 CommonMembraneParameters parameters, WLCParameters wlcParams, KantorBendingParameters kantorParams,
                                 bool stressFree, float growUntil);
    
    ~InteractionMembraneWLCKantor();
};
