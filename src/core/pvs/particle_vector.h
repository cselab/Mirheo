#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/domain.h>

#include "extra_data/extra_data_manager.h"

class ParticleVector;

class LocalParticleVector
{
public:
    ParticleVector* pv;

    PinnedBuffer<Particle> coosvels;
    DeviceBuffer<Force> forces;
    ExtraDataManager extraPerParticle;

    // Local coordinate system; (0,0,0) is center of the local domain
    LocalParticleVector(ParticleVector* pv, int n=0);

    int size() { return np; }
    virtual void resize(const int n, cudaStream_t stream);
    virtual void resize_anew(const int n);
    
    virtual ~LocalParticleVector();

protected:
    int np;
};


class ParticleVector
{
protected:

    LocalParticleVector *_local, *_halo;
    
public:
    
    using PyContainer = std::vector<std::array<float, 3>>;
    
    DomainInfo domain;    

    float mass;
    std::string name;
    // Local coordinate system; (0,0,0) is center of the local domain

    bool haloValid = false;
    bool redistValid = false;

    int cellListStamp{0};

    ParticleVector(std::string name, float mass, int n=0);

    LocalParticleVector* local() { return _local; }
    LocalParticleVector* halo()  { return _halo;  }

    virtual void checkpoint(MPI_Comm comm, std::string path);
    virtual void restart(MPI_Comm comm, std::string path);

    
    // Python getters / setters
    // Use default blocking stream
    std::vector<int> getIndices_vector();
    PyContainer getCoordinates_vector();
    PyContainer getVelocities_vector();
    PyContainer getForces_vector();
    
    void setCoordinates_vector(PyContainer& coordinates);
    void setVelocities_vector(PyContainer& velocities);
    void setForces_vector(PyContainer& forces);
    
    
    virtual ~ParticleVector();
    
    template<typename T>
    void requireDataPerParticle(std::string name, bool needExchange)
    {
        requireDataPerParticle<T>(name, needExchange, 0);
    }
    
    template<typename T>
    void requireDataPerParticle(std::string name, bool needExchange, int shiftDataType)
    {
        requireDataPerParticle<T>(local(), name, needExchange, shiftDataType);
        requireDataPerParticle<T>(halo(),  name, needExchange, shiftDataType);
    }

protected:
    ParticleVector(std::string name, float mass,
                   LocalParticleVector *local, LocalParticleVector *halo );

    int restartIdx = 0;

private:

    template<typename T>
    void requireDataPerParticle(LocalParticleVector* lpv, std::string name, bool needExchange, int shiftDataType)
    {
        lpv->extraPerParticle.createData<T> (name, lpv->size());
        if (needExchange) lpv->extraPerParticle.requireExchange(name);
        if (shiftDataType != 0) lpv->extraPerParticle.requireShift(name, shiftDataType);
    }
};




