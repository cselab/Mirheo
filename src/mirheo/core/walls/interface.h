#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/domain.h>
#include <mirheo/core/mirheo_object.h>
#include <mirheo/core/pvs/particle_vector.h>

#include <mpi.h>
#include <vector>
#include <cuda_runtime.h>

class CellList;
class GPUcontainer;

class Wall : public MirSimulationObject
{
public:
    Wall(const MirState *state, std::string name);
    virtual ~Wall();

    virtual void setup(MPI_Comm& comm) = 0;
    virtual void attachFrozen(ParticleVector *pv) = 0;

    virtual void removeInner(ParticleVector *pv) = 0;
    virtual void attach(ParticleVector *pv, CellList *cl, real maximumPartTravel) = 0;
    virtual void bounce(cudaStream_t stream) = 0;

    /**
     * Ask ParticleVectors which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector *pv);

    virtual void check(cudaStream_t stream) = 0;
};


class SDF_basedWall : public Wall
{
public:
    using Wall::Wall;
    ~SDF_basedWall();
    
    virtual void sdfPerParticle(LocalParticleVector *lpv,
            GPUcontainer *sdfs, GPUcontainer *gradients,
            real gradientThreshold, cudaStream_t stream) = 0;
    virtual void sdfPerPosition(GPUcontainer *positions, GPUcontainer *sdfs, cudaStream_t stream) = 0;
    virtual void sdfOnGrid(real3 gridH, GPUcontainer* sdfs, cudaStream_t stream) = 0;
    virtual PinnedBuffer<double3>* getCurrentBounceForce() = 0;
};
