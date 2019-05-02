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

    ExtraDataManager dataPerObject;

protected:
    int objSize { 0 };

    int getNobjects(int np) const;
};


class ObjectVector : public ParticleVector
{
public:
    
    ObjectVector(const YmrState *state, std::string name, float mass, int objSize, int nObjects = 0);
    virtual ~ObjectVector();
    
    void findExtentAndCOM(cudaStream_t stream, ParticleVectorType type);

    LocalObjectVector* local() { return static_cast<LocalObjectVector*>(ParticleVector::local()); }
    LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(ParticleVector::halo());  }

    void checkpoint (MPI_Comm comm, std::string path, int checkpointId) override;
    void restart    (MPI_Comm comm, std::string path) override;

    template<typename T>
    void requireDataPerObject(std::string name, ExtraDataManager::PersistenceMode persistence)
    {
        requireDataPerObject<T>(name, persistence, 0);
    }

    template<typename T>
    void requireDataPerObject(std::string name, ExtraDataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        requireDataPerObject<T>(local(), name, persistence, shiftDataSize);
        requireDataPerObject<T>(halo(),  name, persistence, shiftDataSize);
    }

public:
    int objSize;
    std::shared_ptr<Mesh> mesh;
    
protected:
    ObjectVector(const YmrState *state, std::string name, float mass, int objSize,
                 std::unique_ptr<LocalParticleVector>&& local,
                 std::unique_ptr<LocalParticleVector>&& halo);

    void _getRestartExchangeMap(MPI_Comm comm, const std::vector<float4>& pos, std::vector<int>& map) override;
    std::vector<int> _restartParticleData(MPI_Comm comm, std::string path) override;

    void _extractPersistentExtraObjectData(std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList = {});
    
    virtual void _checkpointObjectData(MPI_Comm comm, std::string path, int checkpointId);
    virtual void _restartObjectData(MPI_Comm comm, std::string path, const std::vector<int>& map);
    
private:
    template<typename T>
    void requireDataPerObject(LocalObjectVector* lov, std::string name, ExtraDataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        lov->dataPerObject.createData<T> (name, lov->nObjects);
        lov->dataPerObject.setPersistenceMode(name, persistence);
        if (shiftDataSize != 0) lov->dataPerObject.requireShift(name, shiftDataSize);

    }
};




