#pragma once

#include "density.h"
#include "lj.h"
#include "membrane.h"
#include "rod.h"
#include "sdpd.h"

#include <core/utils/pytypes.h>

#include <extern/variant/include/mpark/variant.hpp>
#include <map>
#include <memory>
#include <string>

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

} // namespace InteractionFactory
