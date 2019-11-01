#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

class ParticleVector;

class StationaryWall_Cylinder
{
public:
    enum class Direction {x, y, z};

    StationaryWall_Cylinder(real2 center, real radius, Direction dir, bool inside) :
        center(center), radius(radius), inside(inside)
    {
        switch (dir)
        {
            case Direction::x: _dir = 0; break;
            case Direction::y: _dir = 1; break;
            case Direction::z: _dir = 2; break;
        }
    }

    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain) { this->domain = domain; }

    const StationaryWall_Cylinder& handler() const { return *this; }

    __D__ inline real operator()(real3 coo) const
    {
        real3 gr = domain.local2global(coo);

        real2 projR;
        if (_dir == 0) projR = make_real2(gr.y, gr.z);
        if (_dir == 1) projR = make_real2(gr.x, gr.z);
        if (_dir == 2) projR = make_real2(gr.x, gr.y);

        real dist = math::sqrt(dot(projR-center, projR-center));

        return inside ? dist - radius : radius - dist;
    }

private:
    real2 center;
    real radius;
    int _dir;

    bool inside;

    DomainInfo domain;
};
