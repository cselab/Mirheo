#include "interface.h"

#include <core/utils/common.h>
#include <core/utils/macros.h>

Interaction::Interaction(const MirState *state, std::string name, float rc) :
    MirSimulationObject(state, name),
    rc(rc)
{}

Interaction::~Interaction() = default;

void Interaction::setPrerequisites(__UNUSED ParticleVector *pv1,
                                   __UNUSED ParticleVector *pv2,
                                   __UNUSED CellList *cl1,
                                   __UNUSED CellList *cl2)
{}

std::vector<Interaction::InteractionChannel> Interaction::getIntermediateOutputChannels() const
{
    return {};
}

std::vector<Interaction::InteractionChannel> Interaction::getIntermediateInputChannels() const
{
    return {};
}

std::vector<Interaction::InteractionChannel> Interaction::getFinalOutputChannels() const
{
    return {{ChannelNames::forces, alwaysActive}};
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

const Interaction::ActivePredicate Interaction::alwaysActive = [](){return true;};
