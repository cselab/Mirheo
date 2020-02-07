#pragma once

#include "interface.h"

namespace mirheo
{

class ParticleVector;

/** \brief No forcing term.
    \ingroup Integrators
 */
class ForcingTermNone : public ForcingTerm
{
public:
    void setup(__UNUSED ParticleVector *pv, __UNUSED real t) override
    {}

    __D__ inline real3 operator()(real3 original, __UNUSED Particle p) const override
    {
        return original;
    }
};

} // namespace mirheo
