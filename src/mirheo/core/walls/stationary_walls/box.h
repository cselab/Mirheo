#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

class ParticleVector;

class StationaryWall_Box
{
public:
    StationaryWall_Box(real3 lo, real3 hi, bool inside) :
        lo(lo),
        hi(hi),
        inside(inside)
    {}

    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain) { this->domain = domain; }

    const StationaryWall_Box& handler() const { return *this; }

    __D__ inline real operator()(real3 coo) const
    {
        const real3 gr = domain.local2global(coo);

        const real3 dist3 = math::min(math::abs(gr - lo), math::abs(hi - gr));
        const real dist = math::min(dist3.x, math::min(dist3.y, dist3.z));

        real sign = 1.0_r;
        if (lo.x < gr.x && gr.x < hi.x  &&
            lo.y < gr.y && gr.y < hi.y  &&
            lo.z < gr.z && gr.z < hi.z)
            sign = -1.0_r;

        return inside ? sign*dist : -sign*dist;
    }

private:
    real3 lo, hi;
    bool inside;

    DomainInfo domain;
};
