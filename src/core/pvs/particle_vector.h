#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/domain.h>
#include <core/pvs/data_manager.h>
#include <core/utils/pytypes.h>
#include <core/mirheo_object.h>

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

    PinnedBuffer<Force>& forces();
    PinnedBuffer<float4>& positions();
    PinnedBuffer<float4>& velocities();

    virtual void computeGlobalIds(MPI_Comm comm, cudaStream_t stream);
    
public:
    ParticleVector *pv;
    DataManager dataPerParticle;

protected:
    int np;
};


class ParticleVector : public MirSimulationObject
{
public:
    
    ParticleVector(const MirState *state, std::string name, float mass, int n=0);
    ~ParticleVector() override;
    
    LocalParticleVector* local() { return _local.get(); }
    LocalParticleVector* halo()  { return _halo.get();  }

    const LocalParticleVector* local() const { return _local.get(); }
    const LocalParticleVector* halo()  const { return  _halo.get(); }

    void checkpoint(MPI_Comm comm, std::string path, int checkpointId) override;
    void restart(MPI_Comm comm, std::string path) override;

    
    // Python getters / setters
    // Use default blocking stream
    std::vector<int64_t> getIndices_vector();
    PyTypes::VectorOfFloat3 getCoordinates_vector();
    PyTypes::VectorOfFloat3 getVelocities_vector();
    PyTypes::VectorOfFloat3 getForces_vector();
    
    void setCoosVels_globally(PyTypes::VectorOfFloat6& coosvels, cudaStream_t stream=0);

    void setCoordinates_vector(PyTypes::VectorOfFloat3& coordinates);
    void setVelocities_vector(PyTypes::VectorOfFloat3& velocities);
    void setForces_vector(PyTypes::VectorOfFloat3& forces);    
    
    template<typename T>
    void requireDataPerParticle(std::string name, DataManager::PersistenceMode persistence,
                                DataManager::ShiftMode shift = DataManager::ShiftMode::None)
    {
        requireDataPerParticle<T>(local(), name, persistence, shift);
        requireDataPerParticle<T>(halo(),  name, persistence, shift);
    }

protected:
    ParticleVector(const MirState *state, std::string name, float mass,
                   std::unique_ptr<LocalParticleVector>&& local,
                   std::unique_ptr<LocalParticleVector>&& halo );

    virtual void _getRestartExchangeMap(MPI_Comm comm, const std::vector<float4> &parts, std::vector<int>& map);
    std::vector<int> _redistributeParticleData(MPI_Comm comm, int chunkSize = 1);
    
    void _extractPersistentExtraData(const DataManager& extraData,
                                     std::vector<XDMF::Channel>& channels,
                                     const std::set<std::string>& blackList) const;
    void _extractPersistentExtraParticleData(std::vector<XDMF::Channel>& channels,
                                             const std::set<std::string>& blackList = {}) const;
    
    virtual void _checkpointParticleData(MPI_Comm comm, std::string path, int checkpointId);
    virtual std::vector<int> _restartParticleData(MPI_Comm comm, std::string path);

private:

    template<typename T>
    void requireDataPerParticle(LocalParticleVector *lpv, std::string name,
                                DataManager::PersistenceMode persistence,
                                DataManager::ShiftMode shift)
    {
        lpv->dataPerParticle.createData<T> (name, lpv->size());
        lpv->dataPerParticle.setPersistenceMode(name, persistence);
        lpv->dataPerParticle.setShiftMode(name, shift);
    }

public:    
    float mass;

    bool haloValid   {false};
    bool redistValid {false};

    int cellListStamp{0};

protected:
    std::unique_ptr<LocalParticleVector> _local, _halo;
};




