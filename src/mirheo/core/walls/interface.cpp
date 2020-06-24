// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"
#include <mirheo/core/utils/config.h>

namespace mirheo
{

Wall::Wall(const MirState *state, const std::string& name) :
    MirSimulationObject(state, name)
{}

void Wall::setPrerequisites(__UNUSED ParticleVector *pv)
{}

ConfigObject Wall::_saveSnapshot(Saver& saver, const std::string &typeName)
{
    return MirSimulationObject::_saveSnapshot(saver, "Wall", typeName);
}

} // namespace mirheo
