#pragma once

#include <mpi.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include <core/domain.h>

class LocalParticleVector;
class ParticleVector;
class CellList;
class GPUcontainer;

class Wall
{
public:
    std::string name;

    Wall(std::string name) : name(name) {};

    virtual void setup(MPI_Comm& comm, DomainInfo domain) = 0;
    virtual void attachFrozen(ParticleVector* pv) = 0;

    virtual void removeInner(ParticleVector* pv) = 0;
    virtual void attach(ParticleVector* pv, CellList* cl) = 0;
    virtual void bounce(float dt, cudaStream_t stream) = 0;

    /**
     * Ask ParticleVectors which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector* pv) {}

    virtual void check(cudaStream_t stream) = 0;
    
    /// Save handler state
    virtual void checkpoint(MPI_Comm& comm, std::string path) {}
    /// Restore handler state
    virtual void restart(MPI_Comm& comm, std::string path) {}

    virtual ~Wall() = default;
};


class SDF_basedWall : public Wall
{
public:
    using Wall::Wall;

    virtual void sdfPerParticle(LocalParticleVector* lpv, GPUcontainer* sdfs, GPUcontainer* gradients, cudaStream_t stream) = 0;
    virtual void sdfOnGrid(float3 gridH, GPUcontainer* sdfs, cudaStream_t stream) = 0;


    ~SDF_basedWall() = default;
};
