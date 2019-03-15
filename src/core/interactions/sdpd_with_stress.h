#pragma once

#include "sdpd.h"

class BasicInteractionSDPDWithStress : public BasicInteractionSDPD
{
public:
    ~BasicInteractionSDPDWithStress();
            
protected:
    
    BasicInteractionSDPDWithStress(const YmrState *state, std::string name, float rc,
                                   float viscosity, float kBT, float stressPeriod);
};

template <class PressureEOS, class DensityKernel>
class InteractionSDPDWithStress : public BasicInteractionSDPDWithStress
{
public:
    
    InteractionSDPDWithStress(const YmrState *state, std::string name, float rc,
                              PressureEOS pressure, DensityKernel densityKernel,
                              float viscosity, float kBT, float stressPeriod);

    ~InteractionSDPDWithStress();
};


