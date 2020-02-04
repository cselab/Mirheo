#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

class StationaryWall_Box
{
public:
    StationaryWall_Box(real3 lo, real3 hi, bool inside) :
        lo_(lo),
        hi_(hi),
        inside_(inside)
    {}

    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain)
    {
        domain_ = domain;
    }

    const StationaryWall_Box& handler() const
    {
        return *this;
    }

    __D__ inline real operator()(real3 coo) const
    {
        const real3 gr = domain_.local2global(coo);

        const real3 dist3 = math::min(math::abs(gr - lo_), math::abs(hi_ - gr));
        const real dist = math::min(dist3.x, math::min(dist3.y, dist3.z));

        real sign = 1.0_r;
        if (lo_.x < gr.x && gr.x < hi_.x  &&
            lo_.y < gr.y && gr.y < hi_.y  &&
            lo_.z < gr.z && gr.z < hi_.z)
            sign = -1.0_r;

        return inside_ ? sign * dist : -sign * dist;
    }

private:
    real3 lo_;
    real3 hi_;
    bool inside_;
    DomainInfo domain_;
};

} // namespace mirheo
