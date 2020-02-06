#include "interface.h"

#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

Interaction::Interaction(const MirState *state, std::string name) :
    MirSimulationObject(state, name)
{}

Interaction::~Interaction() = default;

void Interaction::setPrerequisites(__UNUSED ParticleVector *pv1,
                                   __UNUSED ParticleVector *pv2,
                                   __UNUSED CellList *cl1,
                                   __UNUSED CellList *cl2)
{}

std::vector<Interaction::InteractionChannel> Interaction::getInputChannels() const
{
    return {};
}

std::vector<Interaction::InteractionChannel> Interaction::getOutputChannels() const
{
    return {{ChannelNames::forces, alwaysActive}};
}

void Interaction::setState(const MirState *state)
{
    MirSimulationObject::setState(state);
    if (impl) impl->setState(state);
}

bool Interaction::isSelfObjectInteraction() const
{
    return false;
}

void Interaction::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    if (!impl) return;
    impl->checkpoint(comm, path, checkpointId);
}

void Interaction::restart(MPI_Comm comm, const std::string& path)
{
    if (!impl) return;
    impl->restart(comm, path);
}

real Interaction::getCutoffRadius() const
{
    return 1.0_r;
}

const Interaction::ActivePredicate Interaction::alwaysActive = [](){return true;};

} // namespace mirheo
