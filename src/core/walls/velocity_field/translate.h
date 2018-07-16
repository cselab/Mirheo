#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class VelocityField_Translate
{
public:
    VelocityField_Translate(float3 vel) :
        vel(vel)
    {    }

    void setup(MPI_Comm& comm, DomainInfo domain) { }

    const VelocityField_Translate& handler() const { return *this; }

    __D__ inline float3 operator()(float3 coo) const
    {
        return vel;
    }

private:
    float3 vel;

    DomainInfo domain;
};
