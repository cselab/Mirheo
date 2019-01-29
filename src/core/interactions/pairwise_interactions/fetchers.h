#pragma once

#include <core/datatypes.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/helper_math.h>

#ifndef __NVCC__
static float4 readNoCache(const float4* addr)
{
    return *addr;
}
#else
#include <core/utils/cuda_common.h>
#endif


/**
 * fetcher that reads positions only
 */
class ParticleFetcher
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    
    ParticleFetcher(float rc) :
        rc(rc),
        rc2(rc*rc)
    {}

    __D__ inline ParticleType read(const ViewType& view, int id) const
    {
        Particle p;
        p.readCoordinate(view.particles, id);
        return p;
    }

    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return Particle(::readNoCache(view.particles + 2*id), make_float4(0.f, 0.f, 0.f, 0.f));
    }
    
    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const { p.readCoordinate(view.particles, id); }
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { /* no velocity here */ }

    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const
    {
        return distance2(src.r, dst.r) < rc2;
    }
    
protected:

    float rc, rc2;
};

/**
 * fetcher that reads positions and velocities
 */
class ParticleFetcherWithVelocity : public ParticleFetcher
{
public:
    
    ParticleFetcherWithVelocity(float rc) :
        ParticleFetcher(rc)
    {}

    __D__ inline ParticleType read(const ViewType& view, int id) const                     { return  Particle(view.particles, id); }

    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return Particle(::readNoCache(view.particles + 2*id),
                        ::readNoCache(view.particles + 2*id + 1));
    }

    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { p.readVelocity  (view.particles, id); }
};

/**
 * fetcher that reads positions and velocities
 */
class ParticleFetcherWithVelocityAndDensity : public ParticleFetcherWithVelocity
{
public:

    struct ParticleWithDensity
    {
        Particle p;
        float d;
    };

    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensity;
    
    ParticleFetcherWithVelocityAndDensity(float rc) :
        ParticleFetcherWithVelocity(rc)
    {}

    __D__ inline ParticleType read(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::read(view, id),
                view.densities[id]};
    }

    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::readNoCache(view, id),
                view.densities[id]};
    }

    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readCoordinates(p.p, view, id);
    }
    
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readExtraData(p.p, view, id);
        p.d = view.densities[id];
    }

    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const
    {
        return ParticleFetcherWithVelocity::withinCutoff(src.p, dst.p);
    }
};
