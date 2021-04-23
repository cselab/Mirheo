// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"
#include <mirheo/core/utils/config.h>

namespace mirheo
{

ObjectBelongingChecker::ObjectBelongingChecker(const MirState *state, const std::string& name) :
    MirSimulationObject(state, name)
{}

ObjectBelongingChecker::~ObjectBelongingChecker() = default;

std::vector<std::string> ObjectBelongingChecker::getChannelsToBeExchanged() const
{
    return {};
}

ConfigObject ObjectBelongingChecker::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    return MirSimulationObject::_saveSnapshot(saver, "ObjectBelongingChecker", typeName);
}

} // namespace mirheo
