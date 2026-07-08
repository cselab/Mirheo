// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "filters/api.h"
#include "force_kernels/parameters.h"

#include <mirheo/core/mirheo_state.h>

#include <memory>
#include <string>
#include <variant>

namespace mirheo
{

class BaseMembraneInteraction;

/// variant that contains all bending energy parameters
using VarBendingParams = std::variant<KantorBendingParameters, JuelicherBendingParameters>;

/// variant that contains all shear energy parameters
using VarShearParams   = std::variant<WLCParameters, LimParameters>;

/** \brief Construct a MembraneInteraction from parameters
    \param [in] state The global state of the system
    \param [in] name Name of the interaction
    \param [in] commonParams Parameters that are common to all membrane interactions
    \param [in] varBendingParams Bending energy parameters. The interaction will have the corresponding type.
    \param [in] varShearParams Shear energy parameters. The interaction will have the corresponding type.
    \param [in] stressFree \c true if stress free mesh should be employed, \c false otherwise.
    \param [in] initLengthFraction Initial length scale of the parameters, will linearly increase up to 1 after \p growUntil time
    \param [in] growUntil Time interval during which the parameters will be linearly scaled in length
    \param [in] varFilter The filter kernel
    \return A MembraneInteraction with template parameters corresponding to all above variants
 */
std::shared_ptr<BaseMembraneInteraction>
createInteractionMembrane(const MirState *state, const std::string& name,
                          CommonMembraneParameters commonParams,
                          VarBendingParams varBendingParams, VarShearParams varShearParams,
                          bool stressFree, real initLengthFraction, real growUntil, VarMembraneFilter varFilter);

} // namespace mirheo
