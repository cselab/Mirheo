#pragma once

#include "parameters_wrap.h"

#include <mirheo/core/mirheo_state.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

class MembraneInteraction;
class RodInteraction;
class PairwiseInteraction;
class ObjectRodBindingInteraction;

namespace InteractionFactory
{
using VarParam = ParametersWrap::VarParam;
using MapParams = ParametersWrap::MapParams;

std::shared_ptr<PairwiseInteraction>
createPairwiseInteraction(const MirState *state, std::string name, real rc, const std::string type, const MapParams& parameters);


std::shared_ptr<MembraneInteraction>
createInteractionMembrane(const MirState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc,
                          const MapParams& parameters,
                          bool stressFree, real growUntil);

std::shared_ptr<RodInteraction>
createInteractionRod(const MirState *state, std::string name, std::string stateUpdate,
                     bool saveEnergies, const MapParams& parameters);

std::shared_ptr<ObjectRodBindingInteraction>
createInteractionObjRodBinding(const MirState *state, std::string name,
                               real torque, real3 relAnchor, real kBound);

} // namespace InteractionFactory
