#include "sub_step.h"

#include "forcing_terms/none.h"
#include "vv.h"

#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/config.h>

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

static std::vector<Interaction*> configToFastForces(Loader& loader, const ConfigArray& array)
{
    std::vector<Interaction*> v(array.size(), nullptr);
    for (size_t i = 0; i < array.size(); ++i) {
        v[i] = loader.getContext().get<Interaction>(array[i]).get();
        assert(v[i] != nullptr);
    }
    return v;
}

IntegratorSubStep::IntegratorSubStep(const MirState *state, Loader& loader, const ConfigObject& config) :
    IntegratorSubStep{state,
                      config["name"],
                      config["substeps"],
                      configToFastForces(loader, config["fastForces"].getArray())}
{}

IntegratorSubStep::~IntegratorSubStep() = default;

void IntegratorSubStep::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<IntegratorSubStep>(this, _saveSnapshot(saver, "IntegratorSubStep"));
}

ConfigObject IntegratorSubStep::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = Integrator::_saveSnapshot(saver, typeName);
    config.emplace("fastForces",    saver(fastForces_));
    // IntegratorVV<Forcing_None> is hardcoded and stateless. We cannot easily
    // treat it as a MirObject here because the LoaderContext will construct it
    // as an std::shared_ptr, and here we have an std::unique_ptr.
    // config.emplace("subIntegrator", "IntegratorVV<Forcing_None>");
    config.emplace("substeps",      saver(substeps_));
    return config;
}

void IntegratorSubStep::stage1(__UNUSED ParticleVector *pv, __UNUSED cudaStream_t stream)
{}

void IntegratorSubStep::stage2(ParticleVector *pv, cudaStream_t stream)
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
        
        subIntegrator_->stage2(pv, stream);

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
