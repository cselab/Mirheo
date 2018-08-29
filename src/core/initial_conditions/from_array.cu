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

    int count = 0;
    for (int i = 0; i < pos.size(); ++i) {
        auto r_ = pos[i];
        auto u_ = vel[i];

        auto r = make_float3(r_[0], r_[1], r_[2]);
        auto u = make_float3(u_[0], u_[1], u_[2]);

        if (domain.inSubDomain(r)) {
            
            Particle p(Float3_int(r, 0).toFloat4(),
                       Float3_int(u, 0).toFloat4());

            coovelPtr[count++] = p;
        }
    }

    pv->local()->resize(count, stream);
}

