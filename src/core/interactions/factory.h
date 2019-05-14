#pragma once

#include <core/ymero_state.h>
#include <core/utils/pytypes.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <map>
#include <memory>
#include <string>

class InteractionMembrane;
class InteractionRod;
class BasicInteractionDensity;
class BasicInteractionSDPD;
class InteractionLJ;
class InteractionDPD;
class InteractionMDPD;
class ObjectRodBindingInteraction;

namespace InteractionFactory
{
using VarParam = mpark::variant<float, PyTypes::float2, PyTypes::float3>;

std::shared_ptr<InteractionMembrane>
createInteractionMembrane(const YmrState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc,
                          const std::map<std::string, VarParam>& parameters,
                          bool stressFree, float growUntil);

std::shared_ptr<InteractionRod>
createInteractionRod(const YmrState *state, std::string name,
                     const std::map<std::string, VarParam>& parameters);

std::shared_ptr<BasicInteractionDensity>
createPairwiseDensity(const YmrState *state, std::string name, float rc, const std::string& density);

std::shared_ptr<BasicInteractionSDPD>
createPairwiseSDPD(const YmrState *state, std::string name, float rc, float viscosity, float kBT,
                   const std::string& EOS, const std::string& density, bool stress,
                   const std::map<std::string, VarParam>& parameters);

std::shared_ptr<InteractionLJ>
createPairwiseLJ(const YmrState *state, std::string name, float rc, float epsilon, float sigma, float maxForce,
                 std::string awareMode, bool stress, const std::map<std::string, VarParam>& parameters);

std::shared_ptr<InteractionDPD>
createPairwiseDPD(const YmrState *state, std::string name, float rc, float a, float gamma, float kBT, float power,
                  bool stress, const std::map<std::string, VarParam>& parameters);

std::shared_ptr<InteractionMDPD>
createPairwiseMDPD(const YmrState *state, std::string name, float rc, float rd, float a, float b, float gamma, float kbt,
                   float power, bool stress, const std::map<std::string, VarParam>& parameters);

std::shared_ptr<ObjectRodBindingInteraction>
createInteractionObjRodBinding(const YmrState *state, std::string name,
                               float torque, PyTypes::float3 relAnchor, float kBound);

} // namespace InteractionFactory
