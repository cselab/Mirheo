#pragma once

#include "membrane_Juelicher.h"
#include "membrane/parameters.h"

#include <memory>

class InteractionMembraneLimJuelicher : public InteractionMembraneJuelicher
{
public:
    InteractionMembraneLimJuelicher(const YmrState *state, std::string name,
                                    MembraneParameters parameters,
                                    LimParameters limParams,
                                    JuelicherBendingParameters juelicherParams,
                                    bool stressFree, float growUntil);
    
    ~InteractionMembraneLimJuelicher();
};
