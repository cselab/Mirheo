#pragma once

#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>

class VelocityField_None
{
public:

    void setup(float t, DomainInfo domain) {}

    const VelocityField_None& handler() const { return *this; }

    __D__ inline float3 operator()(float3 coo) const
    {
        return {0.f, 0.f, 0.f};
    }
};
