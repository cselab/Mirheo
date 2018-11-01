#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class VelocityField_Oscillate
{
public:
    VelocityField_Oscillate(float3 vel, float period) :
        vel(vel), period(period)
    {
        if (period <= 0)
            die("Oscillating period should be strictly positive");
    }

    void setup(MPI_Comm& comm, float t, DomainInfo domain)
    {
        cosOmega = cos(2*M_PI * t / period);
    }

    const VelocityField_Oscillate& handler() const { return *this; }

    __D__ inline float3 operator()(float3 coo) const
    {
        return vel * cosOmega;
    }

private:
    float3 vel;
    float period;

    float cosOmega;

    DomainInfo domain;
};
