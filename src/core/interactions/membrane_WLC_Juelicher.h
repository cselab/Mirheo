#pragma once

#include "membrane_Juelicher.h"
#include "membrane/parameters.h"

#include <memory>

class InteractionMembraneWLCJuelicher : public InteractionMembraneJuelicher
{
public:
    InteractionMembraneWLCJuelicher(const YmrState *state, std::string name,
                                    MembraneParameters parameters,
                                    WLCParameters wlcParams,
                                    JuelicherBendingParameters juelicherParams,
                                    bool stressFree, float growUntil);
    
    ~InteractionMembraneWLCJuelicher();
};
