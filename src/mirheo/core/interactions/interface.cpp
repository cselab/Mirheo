// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"

#include <mirheo/core/logger.h>
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

void Interaction::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3,
                                   CellList *cl1, CellList *cl2, __UNUSED CellList *cl3)
{
    if (pv3 != nullptr)
        die("interaction '%s': a third particle vector was given to a two-body interaction", getCName());
    setPrerequisites(pv1, pv2, cl1, cl2);
}

void Interaction::local(ParticleVector *pv1, ParticleVector *pv2, __UNUSED ParticleVector *pv3,
                        CellList *cl1, CellList *cl2, __UNUSED CellList *cl3, cudaStream_t stream)
{
    local(pv1, pv2, cl1, cl2, stream);
}

void Interaction::halo(ParticleVector *pv1, ParticleVector *pv2, __UNUSED ParticleVector *pv3,
                       CellList *cl1, CellList *cl2, __UNUSED CellList *cl3, cudaStream_t stream)
{
    halo(pv1, pv2, cl1, cl2, stream);
}

std::vector<Interaction::InteractionChannel> Interaction::getInputChannels() const
{
    return {};
}

std::vector<Interaction::InteractionChannel> Interaction::getOutputChannels() const
{
    return {{channel_names::forces, alwaysActive}};
}

bool Interaction::isSelfObjectInteraction() const
{
    return false;
}

std::optional<real> Interaction::getCutoffRadius() const
{
    return std::nullopt;
}

const Interaction::ActivePredicate Interaction::alwaysActive = [](){return true;};


} // namespace mirheo
