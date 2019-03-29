#pragma once

#include "particle_vector.h"

#include <core/containers.h>
#include <core/logger.h>
#include <core/mesh/mesh.h>
#include <core/utils/common.h>

class LocalObjectVector: public LocalParticleVector
{
protected:
    int objSize  = 0;

public:
    int nObjects = 0;

    bool comExtentValid = false;

    ExtraDataManager extraPerObject;

    struct __align__(16) COMandExtent
    {
        float3 com, low, high;
    };


    LocalObjectVector(ParticleVector* pv, int objSize, int nObjects = 0) :
        LocalParticleVector(pv, objSize*nObjects), objSize(objSize), nObjects(nObjects)
    {
        if (objSize <= 0)
            die("Object vector should contain at least one particle per object instead of %d", objSize);

        resize_anew(nObjects*objSize);
    }

    void resize(const int np, cudaStream_t stream) override
    {
        if (np % objSize != 0)
            die("Incorrect number of particles in object: given %d, must be a multiple of %d", np, objSize);

        nObjects = np / objSize;
        LocalParticleVector::resize(np, stream);

        extraPerObject.resize(nObjects, stream);
    }

    void resize_anew(const int np) override
    {
        if (np % objSize != 0)
            die("Incorrect number of particles in object");

        nObjects = np / objSize;
        LocalParticleVector::resize_anew(np);

        extraPerObject.resize_anew(nObjects);
    }

    virtual PinnedBuffer<Particle>* getMeshVertices(cudaStream_t stream)
    {
        return &coosvels;
    }

    virtual PinnedBuffer<Particle>* getOldMeshVertices(cudaStream_t stream)
    {
        return extraPerParticle.getData<Particle>(ChannelNames::oldParts);
    }

    virtual DeviceBuffer<Force>* getMeshForces(cudaStream_t stream)
    {
        return &forces;
    }


    virtual ~LocalObjectVector() = default;
};


class ObjectVector : public ParticleVector
{
protected:
    ObjectVector(const YmrState *state, std::string name, float mass, int objSize,
                 std::unique_ptr<LocalParticleVector>&& local,
                 std::unique_ptr<LocalParticleVector>&& halo) :
        ParticleVector(state, name, mass, std::move(local), std::move(halo)),
        objSize(objSize)
    {
        // center of mass and extents are not to be sent around
        // it's cheaper to compute them on site
        requireDataPerObject<LocalObjectVector::COMandExtent>(ChannelNames::comExtents, ExtraDataManager::PersistenceMode::None);

        // object ids must always follow objects
        requireDataPerObject<int>(ChannelNames::globalIds, ExtraDataManager::PersistenceMode::Persistent);
    }

public:
    int objSize;
    std::shared_ptr<Mesh> mesh;

    ObjectVector(const YmrState *state, std::string name, float mass, const int objSize, const int nObjects = 0) :
        ObjectVector( state, name, mass, objSize,
                      std::make_unique<LocalObjectVector>(this, objSize, nObjects),
                      std::make_unique<LocalObjectVector>(this, objSize, 0) )
    {}

    void findExtentAndCOM(cudaStream_t stream, ParticleVectorType type);

    LocalObjectVector* local() { return static_cast<LocalObjectVector*>(ParticleVector::local()); }
    LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(ParticleVector::halo());  }

    void checkpoint (MPI_Comm comm, std::string path) override;
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

    virtual ~ObjectVector() = default;

protected:

    void _getRestartExchangeMap(MPI_Comm comm, const std::vector<Particle> &parts, std::vector<int>& map) override;
    std::vector<int> _restartParticleData(MPI_Comm comm, std::string path) override;

    void _extractPersistentExtraObjectData(std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList = {});
    
    virtual void _checkpointObjectData(MPI_Comm comm, std::string path);
    virtual void _restartObjectData(MPI_Comm comm, std::string path, const std::vector<int>& map);
    
private:
    template<typename T>
    void requireDataPerObject(LocalObjectVector* lov, std::string name, ExtraDataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        lov->extraPerObject.createData<T> (name, lov->nObjects);
        lov->extraPerObject.setPersistenceMode(name, persistence);
        if (shiftDataSize != 0) lov->extraPerObject.requireShift(name, shiftDataSize);

    }
};




