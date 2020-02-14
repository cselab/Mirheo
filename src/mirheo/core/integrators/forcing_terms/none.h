#pragma once

#include "interface.h"

#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class ParticleVector;

/** \brief No forcing term.
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

MIRHEO_MEMBER_VARS_0(ForcingTermNone);

} // namespace mirheo
