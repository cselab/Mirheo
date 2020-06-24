// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "utils/parameters_wrap.h"

#include <mirheo/core/mirheo_state.h>

#include <memory>
#include <string>

namespace mirheo
{

class Interaction;
class BaseMembraneInteraction;
class BaseRodInteraction;
class BasePairwiseInteraction;
class ObjectRodBindingInteraction;

namespace interaction_factory
{
using VarParam = ParametersWrap::VarParam;
using MapParams = ParametersWrap::MapParams;

std::shared_ptr<BasePairwiseInteraction>
createPairwiseInteraction(const MirState *state, std::string name, real rc, const std::string type, const MapParams& parameters);


std::shared_ptr<BaseMembraneInteraction>
createInteractionMembrane(const MirState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc,
                          std::string filterDesc, const MapParams& parameters,
                          bool stressFree);

std::shared_ptr<BaseRodInteraction>
createInteractionRod(const MirState *state, std::string name, std::string stateUpdate,
                     bool saveEnergies, const MapParams& parameters);

std::shared_ptr<ObjectRodBindingInteraction>
createInteractionObjRodBinding(const MirState *state, std::string name,
                               real torque, real3 relAnchor, real kBound);

/** \brief Interaction factory. Instantiate the correct interaction object depending on the snapshot parameters.
    \param [in] state The global state of the system.
    \param [in] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] config The interaction parameters.
 */
std::shared_ptr<Interaction>
loadInteraction(const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace interaction_factory

} // namespace mirheo
