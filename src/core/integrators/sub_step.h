#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/datatypes.h>

class Interaction;

class IntegratorSubStep : Integrator
{
public:
    IntegratorSubStep(const MirState *state, std::string name, int substeps, Interaction *fastForces);
    ~IntegratorSubStep();
    
    void stage1(ParticleVector *pv, cudaStream_t stream) override;
    void stage2(ParticleVector *pv, cudaStream_t stream) override;

    void setPrerequisites(ParticleVector *pv) override;

private:

    Interaction *fastForces; /* interactions (self) called `substeps` times per time step */
    std::unique_ptr<Integrator> subIntegrator;
    MirState subState;
    
    int substeps; /* number of substeps */
    DeviceBuffer<Force> slowForces;
    DeviceBuffer<float4> previousPositions;

    void updateSubState();
};
