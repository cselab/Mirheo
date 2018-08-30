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

    std::vector<Particle> localParticles;

    for (int i = 0; i < pos.size(); ++i) {
        auto r_ = pos[i];
        auto u_ = vel[i];

        auto r = make_float3(r_[0], r_[1], r_[2]);
        auto u = make_float3(u_[0], u_[1], u_[2]);

        if (domain.inSubDomain(r)) {

            r = domain.global2local(r);
            
            Particle p(Float3_int(r, 0).toFloat4(),
                       Float3_int(u, 0).toFloat4());

            localParticles.push_back(p);
        }
    }

    pv->local()->resize_anew(localParticles.size());
    auto coovelPtr = pv->local()->coosvels.hostPtr();
    
    for (int i = 0; i < localParticles.size(); ++i)
        coovelPtr[i] = localParticles[i];

    pv->local()->coosvels.uploadToDevice(stream);
}

