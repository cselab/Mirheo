#pragma once

#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class ParticleVector;

class Forcing_None
{
public:
    void setup(__UNUSED ParticleVector* pv, __UNUSED real t) {}

    __D__ inline real3 operator()(real3 original, __UNUSED Particle p) const
    {
        return original;
    }
};

MIRHEO_MEMBER_VARS_0(Forcing_None);

} // namespace mirheo
