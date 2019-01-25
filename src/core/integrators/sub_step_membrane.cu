#include "sub_step_membrane.h"

#include <core/interactions/membrane.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/common.h>
#include <core/utils/kernel_launch.h>

IntegratorSubStepMembrane::IntegratorSubStepMembrane(const YmrState *state, std::string name, int substeps, Interaction *fastForces) :
    Integrator(state, name), substeps(substeps),
    subIntegrator(new IntegratorVV<Forcing_None>(state, name + "_sub", Forcing_None()))
{
    this->fastForces = dynamic_cast<InteractionMembrane*>(fastForces);
    
    if ( this->fastForces == nullptr )
        die("IntegratorSubStepMembrane expects an interaction of type <InteractionMembrane>.");

    subIntegrator->dt = dt / substeps;
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
    for (int substep = 0; substep < substeps; ++ substep) {

        if (substep != 0)
            pv->local()->forces.copy(slowForces, stream);

        // TODO was , t + substep * dt / substeps
        fastForces->local(pv, pv, nullptr, nullptr, stream);
        
        subIntegrator->stage2(pv, stream);
    }
    
    // restore previous positions into old_particles channel
    pv->local()->extraPerParticle.getData<Particle>(ChannelNames::oldParts)->copy(previousPositions, stream);

    // PV may have changed, invalidate all
    pv->haloValid = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}

void IntegratorSubStepMembrane::setPrerequisites(ParticleVector* pv)
{
    fastForces->setPrerequisites(pv, pv);
}
