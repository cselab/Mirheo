#pragma once

#include <core/mirheo_state.h>
#include <core/utils/pytypes.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

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
using VarParam = mpark::variant<float, std::vector<float>, std::vector<PyTypes::float2>>;

std::shared_ptr<InteractionMembrane>
createInteractionMembrane(const MirState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc,
                          const std::map<std::string, VarParam>& parameters,
                          bool stressFree, float growUntil);

std::shared_ptr<InteractionRod>
createInteractionRod(const MirState *state, std::string name, std::string stateUpdate,
                     bool saveEnergies, const std::map<std::string, VarParam>& parameters);

std::shared_ptr<BasicInteractionDensity>
createPairwiseDensity(const MirState *state, std::string name, float rc, const std::string& density);

std::shared_ptr<BasicInteractionSDPD>
createPairwiseSDPD(const MirState *state, std::string name, float rc, float viscosity, float kBT,
                   const std::string& EOS, const std::string& density, bool stress,
                   const std::map<std::string, VarParam>& parameters);

std::shared_ptr<InteractionLJ>
createPairwiseLJ(const MirState *state, std::string name, float rc, float epsilon, float sigma, float maxForce,
                 std::string awareMode, bool stress, const std::map<std::string, VarParam>& parameters);

std::shared_ptr<InteractionDPD>
createPairwiseDPD(const MirState *state, std::string name, float rc, float a, float gamma, float kBT, float power,
                  bool stress, const std::map<std::string, VarParam>& parameters);

std::shared_ptr<InteractionMDPD>
createPairwiseMDPD(const MirState *state, std::string name, float rc, float rd, float a, float b, float gamma, float kbt,
                   float power, bool stress, const std::map<std::string, VarParam>& parameters);

std::shared_ptr<ObjectRodBindingInteraction>
createInteractionObjRodBinding(const MirState *state, std::string name,
                               float torque, PyTypes::float3 relAnchor, float kBound);

} // namespace InteractionFactory
