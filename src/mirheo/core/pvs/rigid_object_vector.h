#pragma once

#include "object_vector.h"
#include <vector_types.h>

namespace mirheo
{

class LocalRigidObjectVector : public LocalObjectVector
{
public:
    LocalRigidObjectVector(ParticleVector *pv, int objSize, int nObjects = 0);

    PinnedBuffer<real4>* getMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<real4>* getOldMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<Force>* getMeshForces(cudaStream_t stream) override;

    void clearRigidForces(cudaStream_t stream);

private:
    PinnedBuffer<real4> meshVertices_;
    PinnedBuffer<real4> meshOldVertices_;
    PinnedBuffer<Force>  meshForces_;
};

class RigidObjectVector : public ObjectVector
{
public:
    RigidObjectVector(const MirState *state, const std::string& name, real partMass, real3 J, const int objSize,
                      std::shared_ptr<Mesh> mesh, const int nObjects = 0);

    virtual ~RigidObjectVector();

    LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(ParticleVector::local()); }
    LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(ParticleVector::halo());  }
    LocalRigidObjectVector* get(ParticleVectorLocality locality)
    {
        return (locality == ParticleVectorLocality::Local) ? local() : halo();
    }

protected:
    void _checkpointObjectData(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void _restartObjectData   (MPI_Comm comm, const std::string& path, const ExchMapSize& ms) override;

public:
    PinnedBuffer<real4> initialPositions;

    /// Diagonal of the inertia tensor in the principal axes
    /// The axes should be aligned with ox, oy, oz when q = {1 0 0 0}
    real3 J;
};

} // namespace mirheo
