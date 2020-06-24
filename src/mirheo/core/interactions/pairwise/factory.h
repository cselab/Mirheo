// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/parameters.h"
#include <mirheo/core/mirheo_state.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <memory>
#include <string>

namespace mirheo
{

class BasePairwiseInteraction;

/** \brief Create a PairwiseInteraction with appropriate template parameters from parameters variants.
    \param [in] state The global state of the system
    \param [in] name The name of the interaction
    \param [in] rc The cut-off radius
    \param [in] varParams Parameters corresponding to the interaction kernel
    \param [in] varStressParams Parameters that controls the stress computation
    \return An instance of PairwiseInteraction
 */
std::shared_ptr<BasePairwiseInteraction>
createInteractionPairwise(const MirState *state, const std::string& name, real rc,
                          const VarPairwiseParams& varParams, const VarStressParams& varStressParams);

/** \brief Create a PairwiseInteraction with appropriate template parameters from a snapshot.
    \param [in] state The global state of the system
    \param [in] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] config The parameters of the interaction.
    \return An instance of PairwiseInteraction.
 */
std::shared_ptr<BasePairwiseInteraction>
loadInteractionPairwise(const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace mirheo
