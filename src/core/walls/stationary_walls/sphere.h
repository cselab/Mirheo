#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class StationaryWall_Sphere
{
public:
    StationaryWall_Sphere(float3 center, float radius, bool inside) :
        center(center), radius(radius), inside(inside)
    {    }

    void setup(__UNUSED MPI_Comm& comm,     DomainInfo domain) { this->domain = domain; }

    const StationaryWall_Sphere& handler() const { return *this; }

    __D__ inline float operator()(float3 coo) const
    {
        float3 gr = domain.local2global(coo);
        float dist = math::sqrt(dot(gr-center, gr-center));

        return inside ? dist - radius : radius - dist;
    }

private:
    float3 center;
    float radius;

    bool inside;

    DomainInfo domain;
};
