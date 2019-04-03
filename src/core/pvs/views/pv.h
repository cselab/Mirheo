#pragma once

#include "../particle_vector.h"

#include <core/utils/common.h>
#include <core/utils/cuda_common.h>

/**
 * GPU-compatible struct of all the relevant data
 */
struct PVview
{
    int size = 0;
    // protected:
    float4 *particles = nullptr;
    // public:
    float4 *forces = nullptr;

    float mass = 0, invMass = 0;

    PVview(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr)
    {
        if (lpv == nullptr) return;

        size = lpv->size();
        particles = reinterpret_cast<float4*>(lpv->coosvels.devPtr());
        forces    = reinterpret_cast<float4*>(lpv->forces.devPtr());

        mass = pv->mass;
        invMass = 1.0 / mass;
    }

    __HD__ inline float4 readPosition(int id) const
    {
        return particles[2*id];
    }

    __HD__ inline float4 readVelocity(int id) const
    {
        return particles[2*id+1];
    }

    __HD__ inline void readPosition(Particle&p, int id) const
    {
        return p.readCoordinate(particles, id);
    }

    __HD__ inline void readVelocity(Particle&p, int id) const
    {
        return p.readVelocity(particles, id);
    }

    __HD__ inline Particle readParticle(int id) const
    {
        return Particle(particles, id);
    }

    __D__ inline float4 readPositionNoCache(int id) const
    {
        return readNoCache(particles + 2*id);
    }

    __D__ inline Particle readParticleNoCache(int id) const
    {
        return {readNoCache(particles + 2*id + 0),
                readNoCache(particles + 2*id + 1)};
    }

    __HD__ inline void writePosition(int id, const float4& r)
    {
        particles[2*id] = r;
    }

    __HD__ inline void writeVelocity(int id, const float4& u)
    {
        particles[2*id + 1] = u;
    }

    __HD__ inline void writeParticle(int id, const Particle& p)
    {
        particles[2*id]   = p.r2Float4();
        particles[2*id+1] = p.u2Float4();
    }
};


struct PVviewWithOldParticles : public PVview
{
    float4 *old_particles = nullptr;

    PVviewWithOldParticles(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
            old_particles = reinterpret_cast<float4*>( lpv->extraPerParticle.getData<Particle>(ChannelNames::oldParts)->devPtr() );
    }

    __HD__ inline void readOldPosition(Particle&p, int id) const
    {
        return p.readCoordinate(old_particles, id);
    }

    __HD__ inline Particle readOldParticle(int id) const
    {
        return {old_particles, id};
    }
};

struct PVviewWithDensities : public PVview
{
    float *densities = nullptr;

    PVviewWithDensities(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
            densities = lpv->extraPerParticle.getData<float>(ChannelNames::densities)->devPtr();
    }
};

template <typename BasicView> 
struct PVviewWithStresses : public BasicView
{
    Stress *stresses = nullptr;

    PVviewWithStresses(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        BasicView(pv, lpv)
    {
        if (lpv != nullptr)
            stresses = lpv->extraPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();            
    }
};

