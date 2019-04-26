#pragma once

#include "object_vector.h"
#include <core/utils/pytypes.h>


class LocalRigidObjectVector : public LocalObjectVector
{
public:
    LocalRigidObjectVector(ParticleVector* pv, int objSize, int nObjects = 0);

    PinnedBuffer<Particle>* getMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<Particle>* getOldMeshVertices(cudaStream_t stream) override;
    PinnedBuffer<Force>* getMeshForces(cudaStream_t stream) override;

protected:
    PinnedBuffer<Particle> meshVertices;
    PinnedBuffer<Particle> meshOldVertices;
    PinnedBuffer<Force>    meshForces;
};

class RigidObjectVector : public ObjectVector
{
public:
    RigidObjectVector(const YmrState *state, std::string name, float partMass, PyTypes::float3 J, const int objSize,
                      std::shared_ptr<Mesh> mesh, const int nObjects = 0);

    virtual ~RigidObjectVector();

    LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(ParticleVector::local()); }
    LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(ParticleVector::halo());  }

protected:
    RigidObjectVector(const YmrState *state, std::string name, float partMass, float3 J, const int objSize,
                      std::shared_ptr<Mesh> mesh, const int nObjects = 0);

    void _checkpointObjectData(MPI_Comm comm, std::string path) override;
    void _restartObjectData(MPI_Comm comm, std::string path, const std::vector<int>& map) override;

public:
    PinnedBuffer<float4> initialPositions;

    /// Diagonal of the inertia tensor in the principal axes
    /// The axes should be aligned with ox, oy, oz when q = {1 0 0 0}
    float3 J;

};




