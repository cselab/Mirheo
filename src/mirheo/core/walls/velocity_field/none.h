// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{
/// Zero velocity field
class VelocityFieldNone
{
public:

    /// to fit the interface
    void setup(__UNUSED real t, __UNUSED DomainInfo domain)
    {}

    /// get a handler that can be used on device
    const VelocityFieldNone& handler() const
    {
        return *this;
    }

    /** Evaluate the velocity field at a given position
        \param [in] r The position in local coordinates
        \return The velocity value
     */
    __D__ inline real3 operator()(__UNUSED real3 r) const
    {
        return {0._r, 0._r, 0._r};
    }
};

} // namespace mirheo
