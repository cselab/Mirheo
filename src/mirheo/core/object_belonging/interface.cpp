// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"

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

} // namespace mirheo
