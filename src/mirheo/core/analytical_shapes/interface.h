#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

class AnalyticShape
{
public:
    virtual __HD__ real inOutFunction(real3 r) const = 0;
    virtual __HD__ real3 normal(real3 r) const = 0;

    virtual real3 inertiaTensor(real totalMass) const = 0;
};

} // namespace mirheo
