#pragma once

#include <set>
#include <string>

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/domain.h>
#include <core/utils/pytypes.h>
#include <core/ymero_object.h>

#include "extra_data/extra_data_manager.h"

namespace XDMF {struct Channel;}

class ParticleVector;

enum class ParticleVectorType {
    Local,
    Halo
};

class LocalParticleVector
{
public:
    ParticleVector* pv;

    PinnedBuffer<Particle> coosvels;
    DeviceBuffer<Force> forces;
    ExtraDataManager extraPerParticle;

    LocalParticleVector(ParticleVector* pv, int n=0);

    int size() { return np; }
    virtual void resize(const int n, cudaStream_t stream);
    virtual void resize_anew(const int n);
    
    virtual ~LocalParticleVector();

protected:
    int np;
};


class ParticleVector : public YmrSimulationObject
{
protected:

    LocalParticleVector *_local, *_halo;
    
public:
    
    float mass;

    bool haloValid = false;
    bool redistValid = false;

    int cellListStamp{0};

    ParticleVector(std::string name, const YmrState *state, float mass, int n=0);

    LocalParticleVector* local() { return _local; }
    LocalParticleVector* halo()  { return _halo;  }

    void checkpoint(MPI_Comm comm, std::string path) override;
    void restart(MPI_Comm comm, std::string path) override;

    
    // Python getters / setters
    // Use default blocking stream
    std::vector<int> getIndices_vector();
    PyTypes::VectorOfFloat3 getCoordinates_vector();
    PyTypes::VectorOfFloat3 getVelocities_vector();
    PyTypes::VectorOfFloat3 getForces_vector();
    
    void setCoosVels_globally(PyTypes::VectorOfFloat6& coosvels, cudaStream_t stream=0);
    void createIndicesHost();

    void setCoordinates_vector(PyTypes::VectorOfFloat3& coordinates);
    void setVelocities_vector(PyTypes::VectorOfFloat3& velocities);
    void setForces_vector(PyTypes::VectorOfFloat3& forces);
    
    
    ~ParticleVector() override;
    
    template<typename T>
    void requireDataPerParticle(std::string name, ExtraDataManager::CommunicationMode communication, ExtraDataManager::PersistenceMode persistence)
    {
        requireDataPerParticle<T>(name, communication, persistence, 0);
    }
    
    template<typename T>
    void requireDataPerParticle(std::string name, ExtraDataManager::CommunicationMode communication, ExtraDataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        requireDataPerParticle<T>(local(), name, communication, persistence, shiftDataSize);
        requireDataPerParticle<T>(halo(),  name, communication, persistence, shiftDataSize);
    }

protected:
    ParticleVector(std::string name, const YmrState *state, float mass,
                   LocalParticleVector *local, LocalParticleVector *halo );

    virtual void _getRestartExchangeMap(MPI_Comm comm, const std::vector<Particle> &parts, std::vector<int>& map);

    void _extractPersistentExtraData(ExtraDataManager& extraData, std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList);
    void _extractPersistentExtraParticleData(std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList = {});
    
    virtual void _checkpointParticleData(MPI_Comm comm, std::string path);
    virtual std::vector<int> _restartParticleData(MPI_Comm comm, std::string path);    

    void advanceRestartIdx();
    int restartIdx = 0;

private:

    template<typename T>
    void requireDataPerParticle(LocalParticleVector* lpv, std::string name, ExtraDataManager::CommunicationMode communication,
                                ExtraDataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        lpv->extraPerParticle.createData<T> (name, lpv->size());
        lpv->extraPerParticle.setExchangeMode(name, communication);
        lpv->extraPerParticle.setPersistenceMode(name, persistence);
        if (shiftDataSize != 0) lpv->extraPerParticle.requireShift(name, shiftDataSize);
    }
};




