#include "from_array.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>

FromArrayIC::FromArrayIC(const std::vector<float3>& pos, const std::vector<float3>& vel) :
    pos(pos),
    vel(vel)
{
    if (pos.size() != vel.size())
        die("pos and vel arrays must have the same size");
}

void FromArrayIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    std::vector<float4> positions, velocities;
    auto domain = pv->state->domain;

    const size_t n = pos.size();
    positions .reserve(n);
    velocities.reserve(n);
    
    for (size_t i = 0; i < n; ++i)
    {
        float3 r = pos[i];
        const float3 u = vel[i];

        if (domain.inSubDomain(r)) {

            r = domain.global2local(r);

            Particle p(Float3_int(r, 0).toFloat4(),
                       Float3_int(u, 0).toFloat4());

            positions .push_back(p.r2Float4());
            velocities.push_back(p.u2Float4());
        }
    }

    pv->local()->resize_anew(positions.size());
    std::copy(positions .begin(), positions .end(), pv->local()->positions() .begin());
    std::copy(velocities.begin(), velocities.end(), pv->local()->velocities().begin());
    pv->local()->positions() .uploadToDevice(stream);
    pv->local()->velocities().uploadToDevice(stream);
    pv->local()->computeGlobalIds(comm, stream);
}

