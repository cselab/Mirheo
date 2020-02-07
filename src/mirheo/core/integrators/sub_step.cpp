#include "sub_step.h"

#include "forcing_terms/none.h"
#include "vv.h"

#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/common.h>

#include <memory>

namespace mirheo
{
IntegratorSubStep::IntegratorSubStep(const MirState *state, const std::string& name, int substeps,
                                     const std::vector<Interaction*>& fastForces) :
    Integrator(state, name),
    fastForces_(fastForces),
    subIntegrator_(std::make_unique<IntegratorVV<Forcing_None>> (state, name + "_sub", Forcing_None())),
    subState_(*state),
    substeps_(substeps)
{
    std::string ffNames = "";

    if (fastForces_.size() == 0)
        die("Integrator '%s' needs at least one integration", getCName());
    
    for (auto ff : fastForces_)
    {
        if (!ff->isSelfObjectInteraction())
            die("IntegratorSubStep '%s': expects a self-interaction (given '%s').",
                getCName(), ff->getCName());

        ffNames += "'" + ff->getName() + "' ";
    }

    debug("setup substep integrator '%s' for %d substeps with sub integrator '%s' and fast forces '%s'",
          getCName(), substeps_, subIntegrator_->getCName(), ffNames.c_str());

    updateSubState_();
    
    subIntegrator_->setState(&subState_);
}

IntegratorSubStep::~IntegratorSubStep() = default;

void IntegratorSubStep::execute(ParticleVector *pv, cudaStream_t stream)
{
    // save "slow forces"
    slowForces_.copyFromDevice(pv->local()->forces(), stream);
    
    // save previous positions
    previousPositions_.copyFromDevice(pv->local()->positions(), stream);

    // advance with internal vv integrator

    updateSubState_();

    // save fastForces state and reset it afterwards
    auto *savedStatePtr = fastForces_[0]->getState();

    for (auto& ff : fastForces_)
        ff->setState(&subState_);
    
    for (int substep = 0; substep < substeps_; ++substep)
    {
        if (substep != 0)
            pv->local()->forces().copy(slowForces_, stream);        

        for (auto ff : fastForces_)
            ff->local(pv, pv, nullptr, nullptr, stream);
        
        subIntegrator_->execute(pv, stream);

        subState_.currentTime += subState_.dt;
        subState_.currentStep ++;
    }
    
    // restore previous positions into old_particles channel
    pv->local()->dataPerParticle.getData<real4>(ChannelNames::oldPositions)->copy(previousPositions_, stream);

    // restore state of fastForces
    for (auto& ff : fastForces_)
        ff->setState(savedStatePtr);

    invalidatePV_(pv);
}

void IntegratorSubStep::setPrerequisites(ParticleVector *pv)
{
    // luckily do not need cell lists for self interactions
    for (auto ff : fastForces_)
        ff->setPrerequisites(pv, pv, nullptr, nullptr);
}

void IntegratorSubStep::updateSubState_()
{
    subState_ = *getState();
    subState_.dt = getState()->dt / static_cast<real>(substeps_);
}

} // namespace mirheo
