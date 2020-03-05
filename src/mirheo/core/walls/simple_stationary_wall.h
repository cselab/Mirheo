#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class LocalParticleVector;
class ParticleVector;
class CellList;

/** \brief SDF based Wall with zero velocity boundary conditions.
    \tparam InsideWallChecker Wall shape representation.
 */
template<class InsideWallChecker>
class SimpleStationaryWall : public SDFBasedWall
{
public:
    /** \brief Construct a SimpleStationaryWall object.
        \param [in] state The simulation state.
        \param [in] name The wall name.
        \param [in] insideWallChecker A functor that represents the wall surface (see stationary_walls/).
     */
    SimpleStationaryWall(const MirState *state, const std::string& name, InsideWallChecker&& insideWallChecker);

    /// Load a wall from a given snapshot.
    SimpleStationaryWall(const MirState *state, Loader& loader, const ConfigObject& config);

    ~SimpleStationaryWall();

    void setup(MPI_Comm& comm) override;
    void setPrerequisites(ParticleVector *pv) override;
    
    void attachFrozen(ParticleVector *pv) override;

    void removeInner(ParticleVector *pv) override;
    void attach(ParticleVector *pv, CellList *cl, real maximumPartTravel) override;
    void bounce(cudaStream_t stream) override;
    void check(cudaStream_t stream) override;

    void sdfPerParticle(LocalParticleVector *pv,
                        GPUcontainer *sdfs, GPUcontainer *gradients,
                        real gradientThreshold, cudaStream_t stream) override;
    void sdfPerPosition(GPUcontainer *positions, GPUcontainer *sdfs, cudaStream_t stream) override;
    void sdfOnGrid(real3 gridH, GPUcontainer *sdfs, cudaStream_t stream) override;

    /// get a reference of the wall surfae representation.
    InsideWallChecker& getChecker() { return insideWallChecker_; }

    PinnedBuffer<double3>* getCurrentBounceForce() override;

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);


private:
    ParticleVector *frozen_ {nullptr}; ///< frozen particles attached to the wall
    PinnedBuffer<int> nInside_{1};     ///< number of particles inside (work space)
    
protected:
    InsideWallChecker insideWallChecker_; ///< The wall shape representation

    std::vector<ParticleVector*> particleVectors_; ///< list of pvs that are attached for bounce
    std::vector<CellList*> cellLists_;             ///< list of cell lists corresponding to particleVectors_

    std::vector<DeviceBuffer<int>> boundaryCells_; ///< ids of all cells adjacent to the wall surface
    PinnedBuffer<double3> bounceForce_{1};         ///< total force exerced on the walls via particles bounce
};

/** \brief Load from a snapshot a SimpleStationaryWall with appropriate template parameters.
    \param [in] state The global state of the system.
    \param [in] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] config The parameters of the interaction.
    \return A Wall instance.
 */
std::shared_ptr<Wall>
loadSimpleStationaryWall(const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace mirheo
