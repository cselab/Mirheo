#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/datatypes.h>

class Interaction;

class IntegratorSubStepMembrane : Integrator
{
public:
    IntegratorSubStepMembrane(const YmrState *state, std::string name, int substeps, Interaction *fastForces);
    ~IntegratorSubStepMembrane();
    
    void stage1(ParticleVector *pv, cudaStream_t stream) override;
    void stage2(ParticleVector *pv, cudaStream_t stream) override;

    void setPrerequisites(ParticleVector* pv) override;        

private:

    Interaction *fastForces; /* interactions (self) called `substeps` times per time step */
    std::unique_ptr<Integrator> subIntegrator;
    YmrState subState;
    
    int substeps; /* number of substeps */
    DeviceBuffer<Force> slowForces;
    DeviceBuffer<Particle> previousPositions;

    void updateSubState();
};
