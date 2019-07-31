#pragma once

#include "particle_vector.h"

#include <core/containers.h>
#include <core/logger.h>
#include <core/mesh/mesh.h>
#include <core/utils/common.h>

class LocalObjectVector: public LocalParticleVector
{
public:
    LocalObjectVector(ParticleVector *pv, int objSize, int nObjects = 0);
    virtual ~LocalObjectVector();

    void resize(int np, cudaStream_t stream) override;
    void resize_anew(int np) override;

    void computeGlobalIds(MPI_Comm comm, cudaStream_t stream) override;
    
    virtual PinnedBuffer<float4>* getMeshVertices(cudaStream_t stream);
    virtual PinnedBuffer<float4>* getOldMeshVertices(cudaStream_t stream);
    virtual PinnedBuffer<Force>* getMeshForces(cudaStream_t stream);


public:
    int nObjects { 0 };
    int objSize  { 0 };
    
    DataManager dataPerObject;

protected:

    int getNobjects(int np) const;
};


class ObjectVector : public ParticleVector
{
public:
    
    ObjectVector(const MirState *state, std::string name, float mass, int objSize, int nObjects = 0);
    virtual ~ObjectVector();
    
    void findExtentAndCOM(cudaStream_t stream, ParticleVectorType type);

    LocalObjectVector* local() { return static_cast<LocalObjectVector*>(ParticleVector::local()); }
    LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(ParticleVector::halo());  }

    void checkpoint (MPI_Comm comm, std::string path, int checkpointId) override;
    void restart    (MPI_Comm comm, std::string path) override;

    template<typename T>
    void requireDataPerObject(std::string name, DataManager::PersistenceMode persistence,
                              DataManager::ShiftMode shift = DataManager::ShiftMode::None)
    {
        requireDataPerObject<T>(local(), name, persistence, shift);
        requireDataPerObject<T>(halo(),  name, persistence, shift);
    }

public:
    int objSize;
    std::shared_ptr<Mesh> mesh;
    
protected:
    ObjectVector(const MirState *state, std::string name, float mass, int objSize,
                 std::unique_ptr<LocalParticleVector>&& local,
                 std::unique_ptr<LocalParticleVector>&& halo);
    
    virtual void _checkpointObjectData(MPI_Comm comm, std::string path, int checkpointId);
    virtual void _restartObjectData(MPI_Comm comm, std::string path, const ExchMapSize& ms);
    
private:
    template<typename T>
    void requireDataPerObject(LocalObjectVector* lov, std::string name,
                              DataManager::PersistenceMode persistence,
                              DataManager::ShiftMode shift)
    {
        lov->dataPerObject.createData<T> (name, lov->nObjects);
        lov->dataPerObject.setPersistenceMode(name, persistence);
        lov->dataPerObject.setShiftMode(name, shift);
    }
};




