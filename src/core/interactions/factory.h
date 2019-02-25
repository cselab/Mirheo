#pragma once

#include "membrane.h"

#include <map>
#include <memory>
#include <string>

namespace InteractionFactory
{

std::shared_ptr<InteractionMembrane>
createInteractionMembrane(const YmrState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc,
                          const std::map<std::string, float>& parameters,
                          bool stressFree, float growUntil);

} // namespace InteractionFactory
