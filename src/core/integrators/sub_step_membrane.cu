#include "sub_step_membrane.h"

#include <core/interactions/membrane.new.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/common.h>
#include <core/utils/kernel_launch.h>

IntegratorSubStepMembrane::IntegratorSubStepMembrane(const YmrState *state, std::string name, int substeps, Interaction *fastForces) :
    Integrator(state, name),
    substeps(substeps),
    subIntegrator(new IntegratorVV<Forcing_None>(state, name + "_sub", Forcing_None())),
    fastForces(fastForces),
    subState(*state)
{    
    if ( dynamic_cast<InteractionMembraneNew*>(fastForces) == nullptr )
        die("IntegratorSubStepMembrane '%s': expects an interaction of type <InteractionMembrane>.",
            name.c_str());

    debug("setup substep integrator '%s' for %d substeps with sub integrator '%s' and fast forces '%s'",
          name.c_str(), substeps, subIntegrator->name.c_str(), fastForces->name.c_str());

    updateSubState();
    
    subIntegrator->state = &subState;
}

IntegratorSubStepMembrane::~IntegratorSubStepMembrane() = default;

void IntegratorSubStepMembrane::stage1(ParticleVector *pv, cudaStream_t stream)
{}

void IntegratorSubStepMembrane::stage2(ParticleVector *pv, cudaStream_t stream)
{
    // save "slow forces"
    slowForces.copy(pv->local()->forces, stream);
    
    // save previous positions
    previousPositions.copyFromDevice(pv->local()->coosvels, stream);

    // advance with internal vv integrator

    updateSubState();

    // save fastForces state and reset it afterwards
    auto *savedStatePtr = fastForces->state;
    fastForces->state = &subState;
    
    for (int substep = 0; substep < substeps; ++ substep) {

        if (substep != 0)
            pv->local()->forces.copy(slowForces, stream);        

        fastForces->local(pv, pv, nullptr, nullptr, stream);
        
        subIntegrator->stage2(pv, stream);

        subState.currentTime += subState.dt;
    }
    
    // restore previous positions into old_particles channel
    pv->local()->extraPerParticle.getData<Particle>(ChannelNames::oldParts)->copy(previousPositions, stream);

    // restore state of fastForces
    fastForces->state = savedStatePtr;
    
    // PV may have changed, invalidate all
    pv->haloValid = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}

void IntegratorSubStepMembrane::setPrerequisites(ParticleVector *pv)
{
    // luckily do not need cell lists for membrane interactions
    fastForces->setPrerequisites(pv, pv, nullptr, nullptr);
}

void IntegratorSubStepMembrane::updateSubState()
{
    subState = *state;
    subState.dt = state->dt / substeps;
}
