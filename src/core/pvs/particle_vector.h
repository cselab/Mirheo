#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/domain.h>
#include <core/pvs/extra_data/extra_data_manager.h>
#include <core/utils/make_unique.h>
#include <core/utils/pytypes.h>
#include <core/ymero_object.h>

#include <memory>
#include <set>
#include <string>


namespace XDMF {struct Channel;}

class ParticleVector;

enum class ParticleVectorType {
    Local,
    Halo
};

class LocalParticleVector
{
public:
    LocalParticleVector(ParticleVector *pv, int n = 0);
    virtual ~LocalParticleVector();
    
    int size() { return np; }
    virtual void resize(int n, cudaStream_t stream);
    virtual void resize_anew(int n);    

    virtual void computeGlobalIds(MPI_Comm comm, cudaStream_t stream);
    
public:
    ParticleVector *pv;

    PinnedBuffer<Particle> coosvels;
    DeviceBuffer<Force> forces;
    ExtraDataManager extraPerParticle;

protected:
    int np;
};


class ParticleVector : public YmrSimulationObject
{
public:
    
    ParticleVector(const YmrState *state, std::string name, float mass, int n=0);
    ~ParticleVector() override;
    
    LocalParticleVector* local() { return _local.get(); }
    LocalParticleVector* halo()  { return _halo.get();  }

    void checkpoint(MPI_Comm comm, std::string path, CheckpointIdAdvanceMode checkpointMode) override;
    void restart(MPI_Comm comm, std::string path) override;

    
    // Python getters / setters
    // Use default blocking stream
    std::vector<int64_t> getIndices_vector();
    PyTypes::VectorOfFloat3 getCoordinates_vector();
    PyTypes::VectorOfFloat3 getVelocities_vector();
    PyTypes::VectorOfFloat3 getForces_vector();
    
    void setCoosVels_globally(PyTypes::VectorOfFloat6& coosvels, cudaStream_t stream=0);
    void createIndicesHost();

    void setCoordinates_vector(PyTypes::VectorOfFloat3& coordinates);
    void setVelocities_vector(PyTypes::VectorOfFloat3& velocities);
    void setForces_vector(PyTypes::VectorOfFloat3& forces);    
    
    template<typename T>
    void requireDataPerParticle(std::string name, ExtraDataManager::PersistenceMode persistence)
    {
        requireDataPerParticle<T>(name, persistence, 0);
    }
    
    template<typename T>
    void requireDataPerParticle(std::string name, ExtraDataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        requireDataPerParticle<T>(local(), name, persistence, shiftDataSize);
        requireDataPerParticle<T>(halo(),  name, persistence, shiftDataSize);
    }

protected:
    ParticleVector(const YmrState *state, std::string name, float mass,
                   std::unique_ptr<LocalParticleVector>&& local,
                   std::unique_ptr<LocalParticleVector>&& halo );

    virtual void _getRestartExchangeMap(MPI_Comm comm, const std::vector<Particle> &parts, std::vector<int>& map);

    void _extractPersistentExtraData(ExtraDataManager& extraData, std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList);
    void _extractPersistentExtraParticleData(std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList = {});
    
    virtual void _checkpointParticleData(MPI_Comm comm, std::string path);
    virtual std::vector<int> _restartParticleData(MPI_Comm comm, std::string path);    

private:

    template<typename T>
    void requireDataPerParticle(LocalParticleVector *lpv, std::string name, ExtraDataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        lpv->extraPerParticle.createData<T> (name, lpv->size());
        lpv->extraPerParticle.setPersistenceMode(name, persistence);
        if (shiftDataSize != 0) lpv->extraPerParticle.requireShift(name, shiftDataSize);
    }

public:    
    float mass;

    bool haloValid   {false};
    bool redistValid {false};

    int cellListStamp{0};

protected:
    std::unique_ptr<LocalParticleVector> _local, _halo;
};




