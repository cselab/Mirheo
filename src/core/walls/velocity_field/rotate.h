#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class VelocityField_Rotate
{
public:
    VelocityField_Rotate(real3 omega, real3 center) :
        omega(omega), center(center)
    {}

    void setup(__UNUSED real t, DomainInfo domain) { this->domain = domain; }

    const VelocityField_Rotate& handler() const { return *this; }

    __D__ inline real3 operator()(real3 coo) const
    {
        real3 gr = domain.local2global(coo);

        return cross(omega, gr - center);
    }

private:
    real3 omega, center;

    DomainInfo domain;
};
