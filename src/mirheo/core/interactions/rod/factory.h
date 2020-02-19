#pragma once

#include "kernels/parameters.h"
#include <mirheo/core/mirheo_state.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <memory>
#include <string>

namespace mirheo
{

using VarSpinParams = mpark::variant<StatesParametersNone,
                                     StatesSmoothingParameters,
                                     StatesSpinParameters>;

class BaseRodInteraction;

std::shared_ptr<BaseRodInteraction>
createInteractionRod(const MirState *state, const std::string& name,
                     RodParameters params, VarSpinParams spinParams, bool saveEnergies);

} // namespace mirheo
