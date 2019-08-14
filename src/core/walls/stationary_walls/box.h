#pragma once

#include <core/domain.h>
#include <core/datatypes.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

class StationaryWall_Box
{
public:
    StationaryWall_Box(float3 lo, float3 hi, bool inside) :
        lo(lo), hi(hi), inside(inside)
    {    }

    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain) { this->domain = domain; }

    const StationaryWall_Box& handler() const { return *this; }

    __D__ inline float operator()(float3 coo) const
    {
        float3 gr = domain.local2global(coo);

        float3 dist3 = fminf(fabs(gr - lo), fabs(hi - gr));
        float dist = min(dist3.x, min(dist3.y, dist3.z));

        float sign = 1.0f;
        if (lo.x < gr.x && gr.x < hi.x  &&  lo.y < gr.y && gr.y < hi.y  &&  lo.z < gr.z && gr.z < hi.z)
            sign = -1.0f;

        return inside ? sign*dist : -sign*dist;
    }

private:
    float3 lo, hi;
    bool inside;

    DomainInfo domain;
};
