#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/datatypes.h>

#include "forcing_terms/none.h"
#include "vv.h"

class Interaction;
class InteractionMembrane;

class IntegratorSubStepMembrane : Integrator
{
public:
    void stage1(ParticleVector *pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector *pv, float t, cudaStream_t stream) override;

    IntegratorSubStepMembrane(std::string name, float dt, int substeps, Interaction *fastForces);

private:

    InteractionMembrane *fastForces; /* interactions (self) called `substeps` times per time step */
    int substeps; /* number of substeps */
    DeviceBuffer<Force> slowForces;
    std::unique_ptr<IntegratorVV<Forcing_None>> subIntegrator;
};
