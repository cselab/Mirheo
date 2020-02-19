#pragma once

#include "kernels/parameters.h"
#include <mirheo/core/mirheo_state.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <memory>
#include <string>

namespace mirheo
{

class BasePairwiseInteraction;

std::shared_ptr<BasePairwiseInteraction>
createInteractionPairwise(const MirState *state, const std::string& name, real rc,
                          const VarPairwiseParams& varParams, const VarStressParams& varStressParams);

std::shared_ptr<BasePairwiseInteraction>
loadInteractionPairwise(const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace mirheo
