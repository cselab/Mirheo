#pragma once

#include "bounce_back.h"
#include "bounce_maxwell.h"

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

using VarBounceKernel = mpark::variant<BounceBack,
                                       BounceMaxwell>;

} // namespace mirheo
