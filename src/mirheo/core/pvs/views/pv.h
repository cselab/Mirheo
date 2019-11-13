#pragma once

#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{

class ParticleVector;
class LocalParticleVector;

/**
 * GPU-compatible struct of all the relevant data
 */
struct PVview
{
    PVview(ParticleVector *pv, LocalParticleVector *lpv);

    __HD__ inline real4 readPosition(int id) const
    {
        return positions[id];
    }

    __HD__ inline real4 readVelocity(int id) const
    {
        return velocities[id];
    }

    __HD__ inline void readPosition(Particle&p, int id) const
    {
        p.readCoordinate(positions, id);
    }

    __HD__ inline void readVelocity(Particle&p, int id) const
    {
        p.readVelocity(velocities, id);
    }

    __HD__ inline Particle readParticle(int id) const
    {
        return Particle(positions[id], velocities[id]);
    }

    __D__ inline real4 readPositionNoCache(int id) const
    {
        return readNoCache(positions + id);
    }

    __D__ inline Particle readParticleNoCache(int id) const
    {
        return {readNoCache(positions  + id),
                readNoCache(velocities + id)};
    }

    __HD__ inline void writePosition(int id, const real4& r)
    {
        positions[id] = r;
    }

    __HD__ inline void writeVelocity(int id, const real4& u)
    {
        velocities[id] = u;
    }

    __HD__ inline void writeParticle(int id, const Particle& p)
    {
        positions [id] = p.r2Real4();
        velocities[id] = p.u2Real4();
    }

    int size {0};
    real4 *positions  {nullptr};
    real4 *velocities {nullptr};
    real4 *forces     {nullptr};

    real mass {0._r}, invMass {0._r};
};


struct PVviewWithOldParticles : public PVview
{
    PVviewWithOldParticles(ParticleVector *pv, LocalParticleVector *lpv);

    __HD__ inline real3 readOldPosition(int id) const
    {
        const auto r = oldPositions[id];
        return {r.x, r.y, r.z};
    }

    real4 *oldPositions {nullptr};
};

struct PVviewWithDensities : public PVview
{
    PVviewWithDensities(ParticleVector *pv, LocalParticleVector *lpv);

    real *densities {nullptr};
};

} // namespace mirheo
