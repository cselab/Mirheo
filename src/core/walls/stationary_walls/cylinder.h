#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class StationaryWall_Cylinder
{
public:
    enum class Direction {x, y, z};

    StationaryWall_Cylinder(float2 center, float radius, Direction dir, bool inside) :
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

    __D__ inline float operator()(float3 coo) const
    {
        float3 gr = domain.local2global(coo);

        float2 projR;
        if (_dir == 0) projR = make_float2(gr.y, gr.z);
        if (_dir == 1) projR = make_float2(gr.x, gr.z);
        if (_dir == 2) projR = make_float2(gr.x, gr.y);

        float dist = math::sqrt(dot(projR-center, projR-center));

        return inside ? dist - radius : radius - dist;
    }

private:
    float2 center;
    float radius;
    int _dir;

    bool inside;

    DomainInfo domain;
};
