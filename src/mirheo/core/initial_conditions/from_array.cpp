#include "from_array.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

FromArrayIC::FromArrayIC(const std::vector<real3>& pos, const std::vector<real3>& vel) :
    pos(pos),
    vel(vel)
{
    if (pos.size() != vel.size())
        die("pos and vel arrays must have the same size");
}

void FromArrayIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    std::vector<real4> positions, velocities;
    auto domain = pv->getState()->domain;

    const size_t n = pos.size();
    positions .reserve(n);
    velocities.reserve(n);
    
    for (size_t i = 0; i < n; ++i)
    {
        real3 r = pos[i];
        const real3 u = vel[i];

        if (domain.inSubDomain(r)) {

            r = domain.global2local(r);

            Particle p(Real3_int(r, 0).toReal4(),
                       Real3_int(u, 0).toReal4());

            positions .push_back(p.r2Real4());
            velocities.push_back(p.u2Real4());
        }
    }

    pv->local()->resize_anew(static_cast<int>(positions.size()));
    std::copy(positions .begin(), positions .end(), pv->local()->positions() .begin());
    std::copy(velocities.begin(), velocities.end(), pv->local()->velocities().begin());
    pv->local()->positions() .uploadToDevice(stream);
    pv->local()->velocities().uploadToDevice(stream);
    pv->local()->computeGlobalIds(comm, stream);
}


} // namespace mirheo
