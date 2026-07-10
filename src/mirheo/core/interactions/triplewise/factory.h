// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/parameters.h"

#include <mirheo/core/mirheo_state.h>

#include <memory>
#include <string>

namespace mirheo
{

class BaseTriplewiseInteraction;

/** \brief Create a TriplewiseInteraction with appropriate template parameters from parameters variants.
    \param [in] state The global state of the system
    \param [in] name The name of the interaction
    \param [in] rc The cut-off radius
    \param [in] varParams Parameters corresponding to the interaction kernel
    \return An instance of TriplewiseInteraction
 */
std::shared_ptr<BaseTriplewiseInteraction>
createInteractionTriplewise(const MirState *state, const std::string& name, real rc, const VarTriplewiseParams& varParams);

} // namespace mirheo
