#pragma once

#include "interface.h"

#include <core/containers.h>

class LocalParticleVector;
class ParticleVector;
class CellList;

template<class InsideWallChecker>
class SimpleStationaryWall : public SDF_basedWall
{
public:
    SimpleStationaryWall(std::string name, InsideWallChecker&& insideWallChecker) :
        SDF_basedWall(name), insideWallChecker(std::move(insideWallChecker))
    {    }

    void setup(MPI_Comm& comm, DomainInfo domain) override;
    void attachFrozen(ParticleVector* pv) override;

    void removeInner(ParticleVector* pv) override;
    void attach(ParticleVector* pv, CellList* cl) override;
    void bounce(float dt, cudaStream_t stream) override;
    void check(cudaStream_t stream) override;

    void sdfPerParticle(LocalParticleVector* pv, GPUcontainer* sdfs, GPUcontainer* gradients, cudaStream_t stream) override;
    void sdfOnGrid(float3 gridH, GPUcontainer* sdfs, cudaStream_t stream) override;


    InsideWallChecker& getChecker() { return insideWallChecker; }

    ~SimpleStationaryWall() = default;

protected:
    MPI_Comm wallComm;
    DomainInfo domain;

    InsideWallChecker insideWallChecker;

    ParticleVector* frozen;
    std::vector<ParticleVector*> particleVectors;
    std::vector<CellList*> cellLists;

    std::vector<int> nBounceCalls;

    std::vector<DeviceBuffer<int>*> boundaryCells;
    PinnedBuffer<int> nInside{1};
};
