#pragma once

#include "interface.h"

#include <core/containers.h>

class LocalParticleVector;
class ParticleVector;
class CellList;

class SDF_basedWall : public Wall
{
public:
    using Wall::Wall;

    virtual void sdfPerParticle(LocalParticleVector* pv, GPUcontainer* sdfs, GPUcontainer* gradients, cudaStream_t stream) = 0;

    ~SDF_basedWall() = default;
};


template<class InsideWallChecker>
class SimpleStationaryWall : public SDF_basedWall
{
public:
    SimpleStationaryWall(std::string name, InsideWallChecker&& insideWallChecker) :
        SDF_basedWall(name), insideWallChecker(std::move(insideWallChecker))
    {    }

    void setup(MPI_Comm& comm, DomainInfo domain, ParticleVector* jointPV) override;

    void removeInner(ParticleVector* pv) override;
    void attach(ParticleVector* pv, CellList* cl) override;
    void bounce(float dt, cudaStream_t stream) override;
    void check(cudaStream_t stream) override;

    void sdfPerParticle(LocalParticleVector* pv, GPUcontainer* sdfs, GPUcontainer* gradients, cudaStream_t stream) override;

    InsideWallChecker& getChecker() { return insideWallChecker; }

    ~SimpleStationaryWall() = default;

protected:
    MPI_Comm wallComm;

    InsideWallChecker insideWallChecker;

    std::vector<ParticleVector*> particleVectors;
    std::vector<CellList*> cellLists;

    std::vector<int> nBounceCalls;

    std::vector<DeviceBuffer<int>*> boundaryCells;
    PinnedBuffer<int> nInside{1};
};
