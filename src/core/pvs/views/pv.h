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
    float4 *positions = nullptr;
    float4 *velocities = nullptr;
    float4 *forces = nullptr;

    float mass = 0, invMass = 0;

    PVview(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr)
    {
        if (lpv == nullptr) return;

        size = lpv->size();
        positions  = lpv->positions() .devPtr();
        velocities = lpv->velocities().devPtr();
        forces     = reinterpret_cast<float4*>(lpv->forces().devPtr());

        mass = pv->mass;
        invMass = 1.0 / mass;
    }

    __HD__ inline float4 readPosition(int id) const
    {
        return positions[id];
    }

    __HD__ inline float4 readVelocity(int id) const
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

    __D__ inline float4 readPositionNoCache(int id) const
    {
        return readNoCache(positions + id);
    }

    __D__ inline Particle readParticleNoCache(int id) const
    {
        return {readNoCache(positions  + id),
                readNoCache(velocities + id)};
    }

    __HD__ inline void writePosition(int id, const float4& r)
    {
        positions[id] = r;
    }

    __HD__ inline void writeVelocity(int id, const float4& u)
    {
        velocities[id] = u;
    }

    __HD__ inline void writeParticle(int id, const Particle& p)
    {
        positions [id] = p.r2Float4();
        velocities[id] = p.u2Float4();
    }
};


struct PVviewWithOldParticles : public PVview
{
    float4 *oldPositions = nullptr;

    PVviewWithOldParticles(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
            oldPositions = lpv->dataPerParticle.getData<float4>(ChannelNames::oldPositions)->devPtr();
    }

    __HD__ inline float3 readOldPosition(int id) const
    {
        const auto r = oldPositions[id];
        return {r.x, r.y, r.z};
    }
};

struct PVviewWithDensities : public PVview
{
    float *densities = nullptr;

    PVviewWithDensities(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
            densities = lpv->dataPerParticle.getData<float>(ChannelNames::densities)->devPtr();
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
            stresses = lpv->dataPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();            
    }
};

