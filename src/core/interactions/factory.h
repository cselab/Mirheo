#pragma once

#include "parameters_wrap.h"

#include <core/mirheo_state.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

class InteractionMembrane;
class InteractionRod;
class PairwiseInteraction;
class ObjectRodBindingInteraction;

namespace InteractionFactory
{
using VarParam = ParametersWrap::VarParam;
using MapParams = ParametersWrap::MapParams;

std::shared_ptr<PairwiseInteraction>
createPairwiseInteraction(const MirState *state, std::string name, float rc, const std::string type, const MapParams& parameters);


std::shared_ptr<InteractionMembrane>
createInteractionMembrane(const MirState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc,
                          const MapParams& parameters,
                          bool stressFree, float growUntil);

std::shared_ptr<InteractionRod>
createInteractionRod(const MirState *state, std::string name, std::string stateUpdate,
                     bool saveEnergies, const MapParams& parameters);

std::shared_ptr<ObjectRodBindingInteraction>
createInteractionObjRodBinding(const MirState *state, std::string name,
                               float torque, PyTypes::float3 relAnchor, float kBound);

} // namespace InteractionFactory
