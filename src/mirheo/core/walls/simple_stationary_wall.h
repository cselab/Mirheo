#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class LocalParticleVector;
class ParticleVector;
class CellList;

template<class InsideWallChecker>
class SimpleStationaryWall : public SDF_basedWall
{
public:

    SimpleStationaryWall(std::string name, const MirState *state, InsideWallChecker&& insideWallChecker);
    ~SimpleStationaryWall();

    void setup(MPI_Comm& comm) override;
    void setPrerequisites(ParticleVector *pv) override;
    
    void attachFrozen(ParticleVector *pv) override;

    void removeInner(ParticleVector *pv) override;
    void attach(ParticleVector *pv, CellList *cl, real maximumPartTravel) override;
    void bounce(cudaStream_t stream) override;
    void check(cudaStream_t stream) override;

    void sdfPerParticle(LocalParticleVector *pv,
                        GPUcontainer *sdfs, GPUcontainer *gradients,
                        real gradientThreshold, cudaStream_t stream) override;
    void sdfPerPosition(GPUcontainer *positions, GPUcontainer* sdfs, cudaStream_t stream) override;
    void sdfOnGrid(real3 gridH, GPUcontainer *sdfs, cudaStream_t stream) override;


    InsideWallChecker& getChecker() { return insideWallChecker; }

    PinnedBuffer<double3>* getCurrentBounceForce() override;

protected:

    InsideWallChecker insideWallChecker;

    ParticleVector *frozen;
    std::vector<ParticleVector*> particleVectors;
    std::vector<CellList*> cellLists;

    std::vector<DeviceBuffer<int>> boundaryCells;
    PinnedBuffer<int> nInside{1};
    PinnedBuffer<double3> bounceForce{1};
};

} // namespace mirheo
