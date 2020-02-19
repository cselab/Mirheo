#pragma once

#include "filters/api.h"
#include "force_kernels/parameters.h"

#include <mirheo/core/mirheo_state.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <memory>
#include <string>

namespace mirheo
{

class BaseMembraneInteraction;

using VarBendingParams = mpark::variant<KantorBendingParameters, JuelicherBendingParameters>;
using VarShearParams   = mpark::variant<WLCParameters, LimParameters>;

std::shared_ptr<BaseMembraneInteraction>
createInteractionMembrane(const MirState *state, const std::string& name,
                          CommonMembraneParameters commonParams,
                          VarBendingParams varBendingParams, VarShearParams varShearParams,
                          bool stressFree, real growUntil, VarMembraneFilter varFilter);

std::shared_ptr<BaseMembraneInteraction>
loadInteractionMembrane(const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace mirheo
