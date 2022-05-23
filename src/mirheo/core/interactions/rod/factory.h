// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/parameters.h"
#include <mirheo/core/mirheo_state.h>

#include <memory>
#include <string>
#include <variant>

namespace mirheo
{
/// variant that contains all possible parameters for the polymorphic states transition models
using VarSpinParams = std::variant<StatesParametersNone,
                                   StatesSmoothingParameters,
                                   StatesSpinParameters>;

class BaseRodInteraction;

/** Construct a RodInteraction object with the corresct template parameters
    \param [in] state The global state of the system
    \param [in] name The name of the interaction
    \param [in] params The force parameters
    \param [in] spinParams The polymorphic transition model parameters
    \param [in] saveEnergies If \c true, will compute and store energies of the bisegments
 */
std::shared_ptr<BaseRodInteraction>
createInteractionRod(const MirState *state, const std::string& name,
                     RodParameters params, VarSpinParams spinParams, bool saveEnergies);

} // namespace mirheo
