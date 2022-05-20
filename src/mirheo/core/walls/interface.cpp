// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"

namespace mirheo
{

Wall::Wall(const MirState *state, const std::string& name) :
    MirSimulationObject(state, name)
{}

void Wall::setPrerequisites(__UNUSED ParticleVector *pv)
{}

} // namespace mirheo
