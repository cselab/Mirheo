#pragma once

#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{

class ParticleVector;
class LocalParticleVector;

/** \brief GPU-compatible struct that contains particle basic data

    Contains particle positions, velocities, forces, and mass info.
 */
struct PVview
{
    /** \brief Construct a \c PVview
        \param [in] pv The ParticleVector that the view represents
        \param [in] lpv The LocalParticleVector that the view represents
     */
    PVview(ParticleVector *pv, LocalParticleVector *lpv);

    /// fetch position from given particle index
    __HD__ inline real4 readPosition(int id) const
    {
        return positions[id];
    }

    /// fetch velocity from given particle index
    __HD__ inline real4 readVelocity(int id) const
    {
        return velocities[id];
    }

    /// fetch position from given particle index and store it into p
    __HD__ inline void readPosition(Particle&p, int id) const
    {
        p.readCoordinate(positions, id);
    }

    /// fetch velocity from given particle index and store it into p
    __HD__ inline void readVelocity(Particle&p, int id) const
    {
        p.readVelocity(velocities, id);
    }

    /// fetch particle from given particle index
    __HD__ inline Particle readParticle(int id) const
    {
        return Particle(positions[id], velocities[id]);
    }

    /// fetch position from given particle index without going through the L1/L2 cache
    /// This can be useful to reduce the cache pressure on concurrent kernels
    __D__ inline real4 readPositionNoCache(int id) const
    {
        return readNoCache(positions + id);
    }

    /// fetch particle from given particle index without going through the L1/L2 cache
    /// This can be useful to reduce the cache pressure on concurrent kernels
    __D__ inline Particle readParticleNoCache(int id) const
    {
        return {readNoCache(positions  + id),
                readNoCache(velocities + id)};
    }

    /// Store position at the given particle id
    __HD__ inline void writePosition(int id, const real4& r)
    {
        positions[id] = r;
    }

    /// Store velocity at the given particle id
    __HD__ inline void writeVelocity(int id, const real4& u)
    {
        velocities[id] = u;
    }

    /// Store particle at the given particle id
    __HD__ inline void writeParticle(int id, const Particle& p)
    {
        positions [id] = p.r2Real4();
        velocities[id] = p.u2Real4();
    }

    int size {0}; ///< number of particles
    real4 *positions  {nullptr}; ///< particle positions in local coordinates
    real4 *velocities {nullptr}; ///< particle velocities
    real4 *forces     {nullptr}; ///< particle forces

    real mass {0._r};    ///< mass of one particle
    real invMass {0._r}; ///< 1 / mass
};


/** \brief \c PVview with additionally positions from previous time steps
 */
struct PVviewWithOldParticles : public PVview
{
    /** \brief Construct a PVviewWithOldParticles
        \param [in] pv The ParticleVector that the view represents
        \param [in] lpv The LocalParticleVector that the view represents

        \rst
        .. note::
            if pv does not have old positions channel, this will be ignored and oldPositions will be set to nullptr.
        \endrst
     */
    PVviewWithOldParticles(ParticleVector *pv, LocalParticleVector *lpv);

    /// fetch positions at previous time step
    __HD__ inline real3 readOldPosition(int id) const
    {
        const auto r = oldPositions[id];
        return {r.x, r.y, r.z};
    }

    real4 *oldPositions {nullptr}; ///< particle positions from previous time steps
};

/** \brief \c PVview with additionally densities data
 */
struct PVviewWithDensities : public PVview
{
    /** \brief Construct a PVviewWithOldParticles
        \param [in] pv The ParticleVector that the view represents
        \param [in] lpv The LocalParticleVector that the view represents

        \rst
        .. warning::
            The pv must hold a density channel.
        \endrst
     */
    PVviewWithDensities(ParticleVector *pv, LocalParticleVector *lpv);

    real *densities {nullptr}; ///< particle densities
};

} // namespace mirheo
