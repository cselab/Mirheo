#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class VelocityField_Translate
{
public:
    VelocityField_Translate(real3 vel) :
        vel(vel)
    {}

    void setup(__UNUSED real t, __UNUSED DomainInfo domain) {}

    const VelocityField_Translate& handler() const { return *this; }

    __D__ inline real3 operator()(__UNUSED real3 coo) const
    {
        return vel;
    }

private:
    real3 vel;

    DomainInfo domain;
};
