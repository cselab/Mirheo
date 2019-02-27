#include "interface.h"

#include <core/utils/common.h>

Interaction::Interaction(const YmrState *state, std::string name, float rc) :
    YmrSimulationObject(state, name),
    rc(rc)
{}

Interaction::~Interaction() = default;

void Interaction::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{}

bool Interaction::outputsForces() const
{
    return true;
}

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

const Interaction::ActivePredicate Interaction::alwaysActive = [](){return true;};
