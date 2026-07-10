// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/parameters.h"

#include <mirheo/core/interactions/utils/parameters_wrap.h>

namespace mirheo
{

namespace factory_helper
{

SW3Params readSW3Params(ParametersWrap &params);

DummyParams readDummyParams(ParametersWrap &params);


} // namespace factory_helper

} // namespace mirheo
