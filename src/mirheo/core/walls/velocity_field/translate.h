#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

class VelocityField_Translate
{
public:
    VelocityField_Translate(real3 vel) :
        vel_(vel)
    {}

    void setup(__UNUSED real t, __UNUSED DomainInfo domain)
    {}

    const VelocityField_Translate& handler() const
    {
        return *this;
    }

    __D__ inline real3 operator()(__UNUSED real3 coo) const
    {
        return vel_;
    }

private:
    real3 vel_;
    DomainInfo domain_;
};

} // namespace mirheo
