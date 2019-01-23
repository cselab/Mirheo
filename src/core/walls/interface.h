#pragma once

#include <mpi.h>
#include <vector>
#include <cuda_runtime.h>
#include <core/domain.h>
#include <core/pvs/particle_vector.h>

#include "core/ymero_object.h"

class CellList;
class GPUcontainer;

class Wall : public YmrSimulationObject
{
public:
    Wall(const YmrState *state, std::string name);
    
    virtual ~Wall();

    virtual void setup(MPI_Comm& comm) = 0;
    virtual void attachFrozen(ParticleVector* pv) = 0;

    virtual void removeInner(ParticleVector* pv) = 0;
    virtual void attach(ParticleVector* pv, CellList* cl) = 0;
    virtual void bounce(cudaStream_t stream) = 0;

    /**
     * Ask ParticleVectors which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector* pv) {}

    virtual void check(cudaStream_t stream) = 0;
};


class SDF_basedWall : public Wall
{
public:
    using Wall::Wall;

    virtual void sdfPerParticle(LocalParticleVector* lpv,
            GPUcontainer* sdfs, GPUcontainer* gradients,
            float gradientThreshold, cudaStream_t stream) = 0;
    virtual void sdfPerPosition(GPUcontainer *positions, GPUcontainer* sdfs, cudaStream_t stream) = 0;
    virtual void sdfOnGrid(float3 gridH, GPUcontainer* sdfs, cudaStream_t stream) = 0;


    ~SDF_basedWall();
};
