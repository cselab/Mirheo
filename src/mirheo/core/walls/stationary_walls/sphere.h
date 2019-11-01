#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

class ParticleVector;

class StationaryWall_Sphere
{
public:
    StationaryWall_Sphere(real3 center, real radius, bool inside) :
        center(center), radius(radius), inside(inside)
    {    }

    void setup(__UNUSED MPI_Comm& comm,     DomainInfo domain) { this->domain = domain; }

    const StationaryWall_Sphere& handler() const { return *this; }

    __D__ inline real operator()(real3 coo) const
    {
        real3 gr = domain.local2global(coo);
        real dist = math::sqrt(dot(gr-center, gr-center));

        return inside ? dist - radius : radius - dist;
    }

private:
    real3 center;
    real radius;

    bool inside;

    DomainInfo domain;
};
