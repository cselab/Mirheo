#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/// fetch positions only
class ParticleFetcher
{
public:
    using ViewType     = PVview;   ///< compatible view type
    using ParticleType = Particle; ///< compatible particle type

    /// \param rc cut-off radius
    ParticleFetcher(real rc) :
        rc_(rc),
        rc2_(rc*rc)
    {}

    /** \brief fetch the particle information
        \param view The view pointing to the data
        \param id The particle index
        \return Particle information
    */
    __D__ inline ParticleType read(const ViewType& view, int id) const
    {
        Particle p;
        readCoordinates(p, view, id);
        return p;
    }

    /** read particle information directly from the global memory (without going through the L1 or L2 cache)
        This may be beneficial if one want to maximize the cahe usage on a concurrent stream
    */
    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return Particle(view.readPositionNoCache(id),
                        make_real4(0._r));
    }

    /// read the coordinates only (used for the first pass on the neighbors, discard cut-off radius)
    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const { view.readPosition(p, id); }
    /// read the additional data contained in the particle (other than coordinates)
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { /* no velocity here */ }

    /// \return \c true if the particles \p src and \p dst are within the cut-off radius distance; \c false otherwise.
    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const
    {
        return distance2(src.r, dst.r) < rc2_;
    }

    /// Generic converter from the ParticleType type to the common \c real3 coordinates
    __D__ inline real3 getPosition(const ParticleType& p) const {return p.r;}
    
protected:
    real rc_;  ///< cut-off radius
    real rc2_; ///< rc^2
};

/// fetch positions and velocities
class ParticleFetcherWithVelocity : public ParticleFetcher
{
public:
    /// \param rc cut-off radius    
    ParticleFetcherWithVelocity(real rc) :
        ParticleFetcher(rc)
    {}

    /// read full particle information
    __D__ inline ParticleType read       (const ViewType& view, int id) const              { return view.readParticle(id);        }
    /// read full particle information through global memory (no L1 or L2 cache)
    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const              { return view.readParticleNoCache(id);  }
    /// read extra particle information
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { view.readVelocity(p, id); }
};

/// fetch positions, velocities and number densities
class ParticleFetcherWithVelocityAndDensity : public ParticleFetcherWithVelocity
{
public:
    /// contains position, global index, velocity and number density of a particle
    struct ParticleWithDensity
    {
        Particle p; ///< positions, global id, velocity
        real d;     ///< number density
    };

    using ViewType     = PVviewWithDensities; ///< compatible view type
    using ParticleType = ParticleWithDensity; ///< compatible particle type

    /// \param rc cut-off radius    
    ParticleFetcherWithVelocityAndDensity(real rc) :
        ParticleFetcherWithVelocity(rc)
    {}

    /// read full particle information
    __D__ inline ParticleType read(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::read(view, id),
                view.densities[id]};
    }

    /// read full particle information through global memory
    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::readNoCache(view, id),
                view.densities[id]};
    }

    /// read particle coordinates only
    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readCoordinates(p.p, view, id);
    }

    /// read velocity and number density of the particle
    __D__ inline void readExtraData(ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readExtraData(p.p, view, id);
        p.d = view.densities[id];
    }

    /// \return \c true if \p src and \p dst are within a cut-off radius distance; \c false otherwise
    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const
    {
        return ParticleFetcherWithVelocity::withinCutoff(src.p, dst.p);
    }

    /// fetch position from the generic particle structure
    __D__ inline real3 getPosition(const ParticleType& p) const {return p.p.r;}
};

/// fetch that reads positions, velocities, number densities and mass
class ParticleFetcherWithVelocityDensityAndMass : public ParticleFetcherWithVelocity
{
public:
    /// contains position, velocity, global id, number density and mass of a particle
    struct ParticleWithDensityAndMass
    {
        Particle p; ///< position, global id, velocity
        real d;     ///< number density
        real m;     ///< mass
    };

    using ViewType     = PVviewWithDensities;        ///< Compatible view type
    using ParticleType = ParticleWithDensityAndMass; ///< Compatible particle type

    /// \param [in] rc The cut-off radius
    ParticleFetcherWithVelocityDensityAndMass(real rc) :
        ParticleFetcherWithVelocity(rc)
    {}

    /// read full particle information
    __D__ inline ParticleType read(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::read(view, id),
                view.densities[id], view.mass};
    }

    /// read full particle information through global memory
    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const
    {
        return {ParticleFetcherWithVelocity::readNoCache(view, id),
                view.densities[id], view.mass};
    }

    /// read particle coordinates only
    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readCoordinates(p.p, view, id);
    }

    /// read velocity, number density and mass of the particle    
    __D__ inline void readExtraData(ParticleType& p, const ViewType& view, int id) const
    {
        ParticleFetcherWithVelocity::readExtraData(p.p, view, id);
        p.d = view.densities[id];
        p.m = view.mass;
    }

    /// \return \c true if \p src and \p dst are within a cut-off radius distance; \c false otherwise
    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const
    {
        return ParticleFetcherWithVelocity::withinCutoff(src.p, dst.p);
    }

    /// fetch position from the generic particle structure
    __D__ inline real3 getPosition(const ParticleType& p) const {return p.p.r;}
};

} // namespace mirheo
