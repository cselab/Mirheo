#include "sub_step_membrane.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/interactions/membrane.h>


IntegratorSubStepMembrane::IntegratorSubStepMembrane(std::string name, float dt, int substeps, Interaction *fastForces) :
    Integrator(name, dt), substeps(substeps),
    subIntegrator(new IntegratorVV<Forcing_None>(name + "_sub", dt/substeps, Forcing_None()))
{
    if ( !(this->fastForces = dynamic_cast<InteractionMembrane*>(fastForces)) )
        die("IntegratorSubStepMembrane expects an interaction of type <InteractionMembrane>.");
}
    

void IntegratorSubStepMembrane::stage1(ParticleVector *pv, float t, cudaStream_t stream)
{}

void IntegratorSubStepMembrane::stage2(ParticleVector *pv, float t, cudaStream_t stream)
{
    // save "slow forces"
    slowForces.copy(pv->local()->forces, stream);
    
    // save previous positions
    previousPositions.copyFromDevice(pv->local()->coosvels, stream);

    // advance with internal vv integrator
    for (int substep = 0; substep < substeps; ++ substep) {

        if (substep != 0)
            pv->local()->forces.copy(slowForces, stream);

        fastForces->regular(pv, pv, nullptr, nullptr, t + substep * dt / substeps, stream);

        subIntegrator->stage2(pv, t, stream);
    }
    
    // restore previous positions into old_particles channel
    pv->local()->extraPerParticle.getData<Particle>("old_particles")->copy(previousPositions, stream);

    // PV may have changed, invalidate all
    pv->haloValid = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}

void IntegratorSubStepMembrane::setPrerequisites(ParticleVector* pv)
{
    fastForces->setPrerequisites(pv, pv);
}
