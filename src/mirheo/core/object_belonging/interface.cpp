#include "interface.h"

ObjectBelongingChecker::ObjectBelongingChecker(const MirState *state, std::string name) :
    MirSimulationObject(state, name)
{}

ObjectBelongingChecker::~ObjectBelongingChecker() = default;

std::vector<std::string> ObjectBelongingChecker::getChannelsToBeExchanged() const
{
    return {};
}
