#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/**
 * fetcher that reads positions only
 */
class ParticleFetcher
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    
    ParticleFetcher(real rc) :
        rc_(rc),
        rc2_(rc*rc)
    {}

    __D__ inline ParticleType read(const ViewType& view, int id) const
    {
        Particle p;
        readCoordinates(p, view, id);
        return p;
    }

    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return Particle(view.readPositionNoCache(id),
                        make_real4(0._r));
    }
    
    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const { view.readPosition(p, id); }
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { /* no velocity here */ }

    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const
    {
        return distance2(src.r, dst.r) < rc2_;
    }

    __D__ inline real3 getPosition(const ParticleType& p) const {return p.r;}
    
protected:

    real rc_, rc2_;
};

/**
 * fetcher that reads positions and velocities
 */
class ParticleFetcherWithVelocity : public ParticleFetcher
{
public:
    
    ParticleFetcherWithVelocity(real rc) :
        ParticleFetcher(rc)
    {}

    __D__ inline ParticleType read       (const ViewType& view, int id) const              { return view.readParticle(id);        }
    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const              { return view.readParticleNoCache(id);  }
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { view.readVelocity(p, id); }
};

/**
 * fetcher that reads positions, velocities and densities
 */
class ParticleFetcherWithVelocityAndDensity : public ParticleFetcherWithVelocity
{
public:

    struct ParticleWithDensity
    {
        Particle p;
        real d;
    };

    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensity;
    
    ParticleFetcherWithVelocityAndDensity(real rc) :
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

    __D__ inline real3 getPosition(const ParticleType& p) const {return p.p.r;}
};

/**
 * fetcher that reads positions, velocities and mass
 */
class ParticleFetcherWithVelocityDensityAndMass : public ParticleFetcherWithVelocity
{
public:

    struct ParticleWithDensityAndMass
    {
        Particle p;
        real d, m;
    };

    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensityAndMass;
    
    ParticleFetcherWithVelocityDensityAndMass(real rc) :
        ParticleFetcherWithVelocity(rc)
    {}

    __D__ inline ParticleType read(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::read(view, id),
                view.densities[id], view.mass};
    }

    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::readNoCache(view, id),
                view.densities[id], view.mass};
    }

    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readCoordinates(p.p, view, id);
    }
    
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readExtraData(p.p, view, id);
        p.d = view.densities[id];
        p.m = view.mass;
    }

    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const
    {
        return ParticleFetcherWithVelocity::withinCutoff(src.p, dst.p);
    }

    __D__ inline real3 getPosition(const ParticleType& p) const {return p.p.r;}
};

} // namespace mirheo
