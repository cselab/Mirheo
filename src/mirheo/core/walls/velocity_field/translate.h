// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

/// Constant velocity field
class VelocityFieldTranslate
{
public:
    /** Construct a VelocityFieldTranslate
        \param [in] vel The constant velocity
    */
    VelocityFieldTranslate(real3 vel) :
        vel_(vel)
    {}

    /// to fir the interface
    void setup(__UNUSED real t, __UNUSED DomainInfo domain)
    {}

    /// get a handler that can be used on the device
    const VelocityFieldTranslate& handler() const
    {
        return *this;
    }

    /** Evaluate the velocity field at a given position
        \param [in] r The position in local coordinates
        \return The velocity value
    */
    __D__ inline real3 operator()(__UNUSED real3 r) const
    {
        return vel_;
    }

private:
    real3 vel_;
    DomainInfo domain_;
};

} // namespace mirheo
