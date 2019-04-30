#include <core/utils/pytypes.h>
#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>

#include "from_array.h"

FromArrayIC::FromArrayIC(const PyTypes::VectorOfFloat3 &pos, const PyTypes::VectorOfFloat3 &vel) :
    pos(pos), vel(vel)
{
    if (pos.size() != vel.size())
        die("pos and vel arrays must have the same size");
}

void FromArrayIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    std::vector<float4> positions, velocities;
    auto domain = pv->state->domain;

    for (int i = 0; i < pos.size(); ++i) {
        auto r_ = pos[i];
        auto u_ = vel[i];

        auto r = make_float3(r_[0], r_[1], r_[2]);
        auto u = make_float3(u_[0], u_[1], u_[2]);

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

