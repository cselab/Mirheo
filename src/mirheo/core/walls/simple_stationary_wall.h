#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class LocalParticleVector;
class ParticleVector;
class CellList;

template<class InsideWallChecker>
class SimpleStationaryWall : public SDFBasedWall
{
public:

    SimpleStationaryWall(const MirState *state, const std::string& name, InsideWallChecker&& insideWallChecker);
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


    InsideWallChecker& getChecker() { return insideWallChecker_; }

    PinnedBuffer<double3>* getCurrentBounceForce() override;

private:
    ParticleVector *frozen_ {nullptr};
    PinnedBuffer<int> nInside_{1};
    
protected:
    InsideWallChecker insideWallChecker_;

    std::vector<ParticleVector*> particleVectors_;
    std::vector<CellList*> cellLists_;

    std::vector<DeviceBuffer<int>> boundaryCells_;
    PinnedBuffer<double3> bounceForce_{1};    
};

} // namespace mirheo
