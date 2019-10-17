#pragma once

#include "object_vector.h"
#include <vector_types.h>

class LocalRigidObjectVector : public LocalObjectVector
{
public:
    LocalRigidObjectVector(ParticleVector *pv, int objSize, int nObjects = 0);

    PinnedBuffer<float4>* getMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<float4>* getOldMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<Force>* getMeshForces(cudaStream_t stream) override;

protected:
    PinnedBuffer<float4> meshVertices;
    PinnedBuffer<float4> meshOldVertices;
    PinnedBuffer<Force>  meshForces;
};

class RigidObjectVector : public ObjectVector
{
public:
    RigidObjectVector(const MirState *state, std::string name, float partMass, float3 J, const int objSize,
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
    PinnedBuffer<float4> initialPositions;

    /// Diagonal of the inertia tensor in the principal axes
    /// The axes should be aligned with ox, oy, oz when q = {1 0 0 0}
    float3 J;

};




