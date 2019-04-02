#pragma once

#include "interface.h"
#include "rod/parameters.h"

class InteractionRod : public Interaction
{
public:
    InteractionRod(const YmrState *state, std::string name, RodParameters params);
    virtual ~InteractionRod();
};
