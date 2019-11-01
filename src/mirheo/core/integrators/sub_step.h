#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>

namespace mirheo
{

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
    DeviceBuffer<Force> slowForces {};
    DeviceBuffer<real4> previousPositions {};

    void updateSubState();
};

} // namespace mirheo
