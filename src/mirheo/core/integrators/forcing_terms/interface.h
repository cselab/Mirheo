#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

class ParticleVector;

/** \brief Interface for assitional force functors.
    \ingroup Integrators
 */
class ForcingTerm
{
public:
    virtual void setup(ParticleVector *pv, real t) = 0;
    virtual __D__ real3 operator()(real3 original, Particle p) const = 0;
};

} // namespace mirheo
