#pragma once

#include "../particle_vector.h"

#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{

/**
 * GPU-compatible struct of all the relevant data
 */
struct PVview
{
    int size {0};
    real4 *positions  {nullptr};
    real4 *velocities {nullptr};
    real4 *forces     {nullptr};

    real mass {0._r}, invMass {0._r};

    PVview(ParticleVector *pv, LocalParticleVector *lpv)
    {
        size = lpv->size();
        positions  = lpv->positions() .devPtr();
        velocities = lpv->velocities().devPtr();
        forces     = reinterpret_cast<real4*>(lpv->forces().devPtr());

        mass = pv->mass;
        invMass = 1.0 / mass;
    }

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
};


struct PVviewWithOldParticles : public PVview
{
    real4 *oldPositions {nullptr};

    PVviewWithOldParticles(ParticleVector *pv, LocalParticleVector *lpv) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
            oldPositions = lpv->dataPerParticle.getData<real4>(ChannelNames::oldPositions)->devPtr();
    }

    __HD__ inline real3 readOldPosition(int id) const
    {
        const auto r = oldPositions[id];
        return {r.x, r.y, r.z};
    }
};

struct PVviewWithDensities : public PVview
{
    real *densities {nullptr};

    PVviewWithDensities(ParticleVector *pv, LocalParticleVector *lpv) :
        PVview(pv, lpv)
    {
        densities = lpv->dataPerParticle.getData<real>(ChannelNames::densities)->devPtr();
    }
};

template <typename BasicView> 
struct PVviewWithStresses : public BasicView
{
    Stress *stresses {nullptr};

    PVviewWithStresses(ParticleVector *pv, LocalParticleVector *lpv) :
        BasicView(pv, lpv)
    {
        stresses = lpv->dataPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();            
    }
};

} // namespace mirheo
