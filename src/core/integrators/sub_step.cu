#include "sub_step.h"

#include "forcing_terms/none.h"
#include "vv.h"

#include <core/interactions/interface.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/common.h>
#include <core/utils/kernel_launch.h>

#include <memory>

IntegratorSubStep::IntegratorSubStep(const YmrState *state, std::string name, int substeps, Interaction *fastForces) :
    Integrator(state, name),
    substeps(substeps),
    subIntegrator(std::make_unique<IntegratorVV<Forcing_None>> (state, name + "_sub", Forcing_None())),
    fastForces(fastForces),
    subState(*state)
{
    if (!fastForces->isSelfObjectInteraction())
        die("IntegratorSubStep '%s': expects a self-interaction (given '%s').",
            name.c_str(), fastForces->name.c_str());

    debug("setup substep integrator '%s' for %d substeps with sub integrator '%s' and fast forces '%s'",
          name.c_str(), substeps, subIntegrator->name.c_str(), fastForces->name.c_str());

    updateSubState();
    
    subIntegrator->state = &subState;
}

IntegratorSubStep::~IntegratorSubStep() = default;

void IntegratorSubStep::stage1(ParticleVector *pv, cudaStream_t stream)
{}

void IntegratorSubStep::stage2(ParticleVector *pv, cudaStream_t stream)
{
    // save "slow forces"
    slowForces.copyFromDevice(pv->local()->forces(), stream);
    
    // save previous positions
    previousPositions.copyFromDevice(pv->local()->positions(), stream);

    // advance with internal vv integrator

    updateSubState();

    // save fastForces state and reset it afterwards
    auto *savedStatePtr = fastForces->state;
    fastForces->state = &subState;
    
    for (int substep = 0; substep < substeps; ++ substep) {

        if (substep != 0)
            pv->local()->forces().copy(slowForces, stream);        

        fastForces->local(pv, pv, nullptr, nullptr, stream);
        
        subIntegrator->stage2(pv, stream);

        subState.currentTime += subState.dt;
    }
    
    // restore previous positions into old_particles channel
    pv->local()->dataPerParticle.getData<float4>(ChannelNames::oldPositions)->copy(previousPositions, stream);

    // restore state of fastForces
    fastForces->state = savedStatePtr;

    invalidatePV(pv);
}

void IntegratorSubStep::setPrerequisites(ParticleVector *pv)
{
    // luckily do not need cell lists for self interactions
    fastForces->setPrerequisites(pv, pv, nullptr, nullptr);
}

void IntegratorSubStep::updateSubState()
{
    subState = *state;
    subState.dt = state->dt / substeps;
}
