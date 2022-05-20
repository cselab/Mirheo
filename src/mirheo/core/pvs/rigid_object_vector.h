// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "object_vector.h"
#include <vector_types.h>

namespace mirheo
{

/** \brief Rigid objects container.

    This is used to represent local or halo objects in RigidObjectVector.
    A rigid object is composed of frozen particles inside a volume that is represented by a triangle mesh.
    There is then two sets of particles: mesh vertices and frozen particles.
    The frozen particles are stored in the particle data manager, while the mesh particles are stored in additional buffers.

    Additionally, each rigid object has a RigidMotion datum associated that fully describes its state.
*/
class LocalRigidObjectVector : public LocalObjectVector
{
public:
    /** \brief Construct a LocalRigidObjectVector.
        \param [in] pv Parent RigidObjectVector
        \param [in] objSize Number of frozen particles per object
        \param [in] nObjects Number of objects
    */
    LocalRigidObjectVector(ParticleVector *pv, int objSize, int nObjects = 0);

    PinnedBuffer<real4>* getMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<real4>* getOldMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<Force>* getMeshForces(cudaStream_t stream) override;

    /// set forces in rigid motions to zero
    void clearRigidForces(cudaStream_t stream);

private:
    PinnedBuffer<real4> meshVertices_;
    PinnedBuffer<real4> meshOldVertices_;
    PinnedBuffer<Force> meshForces_;
};

/** \brief Rigid objects container.

    Holds two LocalRigidObjectVector: local and halo.
 */
class RigidObjectVector : public ObjectVector
{
public:
    /** Construct a RigidObjectVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] partMass Mass of one frozen particle
        \param [in] J Diagonal entries of the inertia tensor, which must be diagonal.
        \param [in] objSize Number of particles per object
        \param [in] mesh Mesh representing the surface of the object.
        \param [in] nObjects Number of objects
    */
    RigidObjectVector(const MirState *state, const std::string& name, real partMass, real3 J, const int objSize,
                      std::shared_ptr<Mesh> mesh, const int nObjects = 0);

    virtual ~RigidObjectVector();

    /// get local LocalRigidObjectVector
    LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(ParticleVector::local()); }
    /// get halo LocalRigidObjectVector
    LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(ParticleVector::halo());  }
    /// get LocalRigidObjectVector from locality
    LocalRigidObjectVector* get(ParticleVectorLocality locality)
    {
        return (locality == ParticleVectorLocality::Local) ? local() : halo();
    }

    /// get diagonal entries of the inertia tensor
    real3 getInertialTensor() const {return J_;}

protected:

    void _checkpointObjectData(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void _restartObjectData   (MPI_Comm comm, const std::string& path, const ExchMapSize& ms) override;

private:
    void _snapshotObjectData(MPI_Comm comm, const std::string& filename,
                             const std::string& initialPosFilename);

public:
    PinnedBuffer<real4> initialPositions; ///< Coordinates of the frozen particles in the frame of reference of the object

private:
    /** Diagonal of the inertia tensor in the principal axes
        The axes should be aligned with ox, oy, oz when q = {1 0 0 0}
    */
    real3 J_;
};

} // namespace mirheo
