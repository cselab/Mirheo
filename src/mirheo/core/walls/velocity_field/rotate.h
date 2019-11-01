#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

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
