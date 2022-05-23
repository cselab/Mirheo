// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "bounce_back.h"
#include "bounce_maxwell.h"

#include <variant>

namespace mirheo
{

/// a variant that contains one of the bounce kernels
using VarBounceKernel = std::variant<BounceBack,
                                     BounceMaxwell>;

} // namespace mirheo
