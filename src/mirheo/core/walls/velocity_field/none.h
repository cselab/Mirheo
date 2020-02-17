#pragma once

#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

class VelocityFieldNone
{
public:

    void setup(__UNUSED real t, __UNUSED DomainInfo domain)
    {}

    const VelocityFieldNone& handler() const
    {
        return *this;
    }

    __D__ inline real3 operator()(__UNUSED real3 coo) const
    {
        return {0._r, 0._r, 0._r};
    }
};

} // namespace mirheo
