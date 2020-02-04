#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/mirheo_object.h>
#include <mirheo/core/pvs/data_manager.h>
#include <mirheo/core/utils/pytypes.h>

#include <memory>
#include <string>
#include <vector>

namespace mirheo
{

class ParticleVector;

enum class ParticleVectorLocality {
    Local,
    Halo
};

std::string getParticleVectorLocalityStr(ParticleVectorLocality locality);

class LocalParticleVector
{
public:
    LocalParticleVector(ParticleVector *pv, int np = 0);
    virtual ~LocalParticleVector();

    friend void swap(LocalParticleVector &, LocalParticleVector &);
    template <typename T>
    friend void swap(LocalParticleVector &, T &) = delete;  // Disallow implicit upcasts.
    
    int size() const noexcept { return np_; }
    virtual void resize(int n, cudaStream_t stream);
    virtual void resize_anew(int n);    

    PinnedBuffer<Force>& forces();
    PinnedBuffer<real4>& positions();
    PinnedBuffer<real4>& velocities();

    virtual void computeGlobalIds(MPI_Comm comm, cudaStream_t stream);
    
public:
    ParticleVector *pv;
    DataManager dataPerParticle;

private:
    int np_;
};


class ParticleVector : public MirSimulationObject
{
public:
    
    ParticleVector(const MirState *state, const std::string& name, real mass, int n=0);
    ~ParticleVector() override;
    Config getConfig() const override;
    
    LocalParticleVector* local() { return local_.get(); }
    LocalParticleVector* halo()  { return halo_.get();  }
    LocalParticleVector* get(ParticleVectorLocality locality)
    {
        return (locality == ParticleVectorLocality::Local) ? local() : halo();
    }

    const LocalParticleVector* local() const { return local_.get(); }
    const LocalParticleVector* halo()  const { return  halo_.get(); }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

    
    // Python getters / setters
    // Use default blocking stream
    std::vector<int64_t> getIndices_vector();
    PyTypes::VectorOfReal3 getCoordinates_vector();
    PyTypes::VectorOfReal3 getVelocities_vector();
    PyTypes::VectorOfReal3 getForces_vector();
    
    void setCoordinates_vector(const std::vector<real3>& coordinates);
    void setVelocities_vector(const std::vector<real3>& velocities);
    void setForces_vector(const std::vector<real3>& forces);
    
    template<typename T>
    void requireDataPerParticle(const std::string& name, DataManager::PersistenceMode persistence,
                                DataManager::ShiftMode shift = DataManager::ShiftMode::None)
    {
        requireDataPerParticle<T>(local(), name, persistence, shift);
        requireDataPerParticle<T>(halo(),  name, persistence, shift);
    }

protected:
    ParticleVector(const MirState *state, const std::string& name, real mass,
                   std::unique_ptr<LocalParticleVector>&& local,
                   std::unique_ptr<LocalParticleVector>&& halo );

    using ExchMap = std::vector<int>;
    struct ExchMapSize
    {
        ExchMap map;
        int newSize;
    };
    
    virtual void     _checkpointParticleData(MPI_Comm comm, const std::string& path, int checkpointId);
    virtual ExchMapSize _restartParticleData(MPI_Comm comm, const std::string& path, int chunkSize);

private:

    template<typename T>
    void requireDataPerParticle(LocalParticleVector *lpv, const std::string& name,
                                DataManager::PersistenceMode persistence,
                                DataManager::ShiftMode shift)
    {
        lpv->dataPerParticle.createData<T> (name, lpv->size());
        lpv->dataPerParticle.setPersistenceMode(name, persistence);
        lpv->dataPerParticle.setShiftMode(name, shift);
    }

public:    
    real mass;

    bool haloValid   {false};
    bool redistValid {false};

    int cellListStamp{0};

private:
    std::unique_ptr<LocalParticleVector> local_, halo_;
};

} // namespace mirheo
