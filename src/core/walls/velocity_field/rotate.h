#pragma once

#include <core/domain.h>
#include <core/datatypes.h>
#include <mpi.h>

class ParticleVector;

class VelocityField_Rotate
{
public:
    VelocityField_Rotate(float3 omega, float3 center) :
        omega(omega), center(center)
    {    }

    void setup(MPI_Comm& comm, DomainInfo domain) { this->domain = domain; }

    const VelocityField_Rotate& handler() const { return *this; }

    __device__ inline float3 operator()(float3 coo) const
    {
        float3 gr = domain.local2global(coo);

        return cross(omega, gr - center);
    }

private:
    float3 omega, center;

    DomainInfo domain;
};
