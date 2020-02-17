#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

class StationaryWallCylinder
{
public:
    enum class Direction {x, y, z};

    StationaryWallCylinder(real2 center, real radius, Direction dir, bool inside) :
        center_(center),
        radius_(radius),
        dir_(dir),
        inside_(inside)
    {}

    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain)
    {
        domain_ = domain;
    }

    const StationaryWallCylinder& handler() const
    {
        return *this;
    }

    __D__ inline real operator()(real3 coo) const
    {
        const real3 gr = domain_.local2global(coo);

        real2 projR;
        if (dir_ == Direction::x) projR = make_real2(gr.y, gr.z);
        if (dir_ == Direction::y) projR = make_real2(gr.x, gr.z);
        if (dir_ == Direction::z) projR = make_real2(gr.x, gr.y);

        const real2 dr = projR - center_;
        const real dist = math::sqrt(dot(dr, dr));

        return inside_ ? dist - radius_ : radius_ - dist;
    }

private:
    real2 center_;
    real radius_;
    Direction dir_;
    bool inside_;

    DomainInfo domain_;
};

} // namespace mirheo
