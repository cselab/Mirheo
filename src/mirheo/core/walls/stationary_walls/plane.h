#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

class StationaryWall_Plane
{
public:
    StationaryWall_Plane(real3 normal, real3 pointThrough) :
        normal_(normalize(normal)),
        pointThrough_(pointThrough)
    {}

    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain)
    {
        domain_ = domain;
    }

    const StationaryWall_Plane& handler() const
    {
        return *this;
    }

    __D__ inline real operator()(real3 coo) const
    {
        const real3 gr = domain_.local2global(coo);
        const real dist = dot(normal_, gr - pointThrough_);
        return dist;
    }

private:
    real3 normal_;
    real3 pointThrough_;
    DomainInfo domain_;
};

} // namespace mirheo
