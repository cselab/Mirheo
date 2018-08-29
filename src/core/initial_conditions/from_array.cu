#include <core/utils/pytypes.h>
#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>

#include "from_array.h"

FromArrayIC::FromArrayIC(const PyContainer &pos, const PyContainer &vel) :
    pos(pos), vel(vel)
{
    if (pos.size() != vel.size())
        die("pos and vel arrays must have the same size");
}

void FromArrayIC::exec(const MPI_Comm& comm, ParticleVector *pv, DomainInfo domain, cudaStream_t stream)
{
    pv->domain = domain;

    pv->local()->resize_anew(pos.size());
    
    auto coovelPtr = pv->local()->coosvels.hostPtr();

    for (int i = 0; i < pos.size(); ++i) {
        auto r = pos[i];
        auto u = vel[i];

        Particle p(make_float4(r[0], r[1], r[2], 0.f),
                   make_float4(u[0], u[1], u[2], 0.f));

        coovelPtr[i] = p;
    }
}

