// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/parameters.h"

#include <mirheo/core/interactions/utils/parameters_wrap.h>

#include <limits>

namespace mirheo {
namespace factory_helper {

DPDParams                readDPDParams                (ParametersWrap& desc);
LJParams                 readLJParams                 (ParametersWrap& desc);
RepulsiveLJParams        readRepulsiveLJParams        (ParametersWrap& desc);
GrowingRepulsiveLJParams readGrowingRepulsiveLJParams (ParametersWrap& desc);
MorseParams              readMorseParams              (ParametersWrap& desc);
MDPDParams               readMDPDParams               (ParametersWrap& desc);
DensityParams            readDensityParams            (ParametersWrap& desc);
SDPDParams               readSDPDParams               (ParametersWrap& desc);

VarStressParams          readStressParams             (ParametersWrap& desc);


} // factory_helper
} // namespace mirheo
