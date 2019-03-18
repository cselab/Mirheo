#include "interface.h"

ObjectBelongingChecker::ObjectBelongingChecker(const YmrState *state, std::string name) :
    YmrSimulationObject(state, name)
{}

ObjectBelongingChecker::~ObjectBelongingChecker() = default;

std::vector<std::string> ObjectBelongingChecker::getChannelsToBeExchanged() const
{
    return {};
}
