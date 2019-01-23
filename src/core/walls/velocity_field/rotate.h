#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class VelocityField_Rotate
{
public:
    VelocityField_Rotate(float3 omega, float3 center) :
        omega(omega), center(center)
    {}

    void setup(float t, DomainInfo domain) { this->domain = domain; }

    const VelocityField_Rotate& handler() const { return *this; }

    __D__ inline float3 operator()(float3 coo) const
    {
        float3 gr = domain.local2global(coo);

        return cross(omega, gr - center);
    }

private:
    float3 omega, center;

    DomainInfo domain;
};
