#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

class StationaryWall_Sphere
{
public:
    StationaryWall_Sphere(real3 center, real radius, bool inside) :
        center_(center),
        radius_(radius),
        inside_(inside)
    {}

    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain)
    {
        domain_ = domain;
    }

    const StationaryWall_Sphere& handler() const
    {
        return *this;
    }

    __D__ inline real operator()(real3 coo) const
    {
        const real3 dr = domain_.local2global(coo) - center_;
        const real dist = length(dr);

        return inside_ ? dist - radius_ : radius_ - dist;
    }

private:
    real3 center_;
    real radius_;
    bool inside_;
    DomainInfo domain_;
};

} // namespace mirheo
