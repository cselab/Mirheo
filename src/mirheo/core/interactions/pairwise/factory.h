// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/parameters.h"

#include <mirheo/core/mirheo_state.h>
#include <mirheo/core/interactions/utils/parameters_wrap.h>

#include <memory>
#include <optional>
#include <string>

namespace mirheo {

class BasePairwiseInteraction;

/** \brief Create a BasePairwiseInteraction with appropriate template parameters from parameters variants.
    \param [in] state The global state of the system
    \param [in] name The name of the interaction
    \param [in] rc The cut-off radius
    \param [in] kind Type of interaction
    \param [in] desc keyword arguments: map from parameter names to values
    \return An instance of BasePairwiseInteraction
 */
std::unique_ptr<BasePairwiseInteraction>
createInteractionPairwise(const MirState *state, const std::string& name, real rc,
                          const std::string& kind, ParametersWrap& desc);


} // namespace mirheo
