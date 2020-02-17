#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/domain.h>
#include <mirheo/core/mirheo_object.h>
#include <mirheo/core/pvs/particle_vector.h>

#include <mpi.h>
#include <vector>
#include <cuda_runtime.h>

namespace mirheo
{

class CellList;
class GPUcontainer;

/** \brief Physical boundaries of the simulation.

    A wall is composed of its surface.
    Additionally, frozen particles can be created from that surface and attached to the wall.
 */
class Wall : public MirSimulationObject
{
public:
    /** \brief Construct a \c Wall.
        \param [in] state The simulation state.
        \param [in] name The name of the wall.
     */
    Wall(const MirState *state, const std::string& name);
    virtual ~Wall();

    /** \brief Initialize the wall internal state.
        \param [in] comm The MPI carthesian communicator of the simulation.

        This must be called before any other wall operations that involve its surface. 
     */
    virtual void setup(MPI_Comm& comm) = 0;

    /** \brief Set frozen particles to the wall.
        \param [in,out] pv The frozen particles.

        The frozen particles may be modified in the operation (velocities set to the wall's one).
     */
    virtual void attachFrozen(ParticleVector *pv) = 0;

    /** \brief Remove particles inside the walls.
        \param [in,out] pv \c ParticleVector to remove the particles from.
        
        If pv is an \c ObjectVector, any object with at least one particle will be removed by this operation.
    */ 
    virtual void removeInner(ParticleVector *pv) = 0;

    /** \brief Register a \c ParticleVector that needs to be bounced from the wall.
        \param [in] pv The particles to be bounced. Will be ignored if it is the same as the frozen particles.
        \param [in] cl Cell lists corresponding to pv.
        \param [in] maximumPartTravel The estimated maximum distance traveled by one particle over a single time step.
        
        Multiple \c ParticleVector can be registered by calling this method several times.
        The parameter maximumPartTravel is used for performance, lower leading to higher performances.
        Note that if it is too low, some particles may be ignored and not bounced and end up inside the walls (see bounce()).
     */
    virtual void attach(ParticleVector *pv, CellList *cl, real maximumPartTravel) = 0;

    /** \brief Bounce the particles attached to the wall.
        \param [in] stream The stream to execute the bounce operation on.
        
        The particles that are bounced must be registered previously exactly once with attach().
     */
    virtual void bounce(cudaStream_t stream) = 0;

    /** \brief Set properties needed by the particles to be bounced. 
        \param [in,out] pv Particles to add additional properties to.
        
        Must be called just after setup() and before any bounce().
        Default: ask nothing.
     */
    virtual void setPrerequisites(ParticleVector *pv);

    /** \brief Counts number of particles inside the walls and report it in the logs.
        \param [in] stream The stream to execute the check operation on.

        The particles that are counted must be previously attached to the walls by calling attach().
     */
    virtual void check(cudaStream_t stream) = 0;
};

/** \brief \c Wall with surface represented via a signed distance function (SDF).

    The surface of the wall is the zero level set of its SDF.
    The SDF has positive values **outside** the simulation domain (called inside the walls), and is negative **inside** the simulation domain.
 */
class SDFBasedWall : public Wall
{
public:
    using Wall::Wall;
    ~SDFBasedWall();

    /** \brief Compute the wall SDF at particles positions.
        \param [in] lpv Input particles.
        \param [out] sdfs Values of the SDF at the particle positions.
        \param [out] gradients Gradients of the SDF at the particle positions. 
                     Can be disabled by passing a nullptr.
        \param [in] gradientThreshold Compute gradients for particles that are only within that distance. 
                    Irrelevant if gradients is nullptr.
        \param [in] stream The stream to execute the operation on.
     */
    virtual void sdfPerParticle(LocalParticleVector *lpv,
            GPUcontainer *sdfs, GPUcontainer *gradients,
            real gradientThreshold, cudaStream_t stream) = 0;

    /** \brief Compute the wall SDF at given positions.
        \param [in] positions Input positions.
        \param [out] sdfs Values of the SDF at the given positions.
        \param [in] stream The stream to execute the operation on.
     */
    virtual void sdfPerPosition(GPUcontainer *positions, GPUcontainer *sdfs, cudaStream_t stream) = 0;

    /** \brief Compute the wall SDF on a uniform grid.
        \param [in] gridH grid spacing.
        \param [out] sdfs Values of the SDF at the grid nodes positions.
        \param [in] stream The stream to execute the operation on.

        This method will resize the sdfs container internally.
    */
    virtual void sdfOnGrid(real3 gridH, GPUcontainer *sdfs, cudaStream_t stream) = 0;

    /// \brief Get accumulated force of particles on the wall at the previous bounce() operation..
    virtual PinnedBuffer<double3>* getCurrentBounceForce() = 0;
};

} // namespace mirheo
