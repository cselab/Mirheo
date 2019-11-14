#pragma once

#include "type_list.h"

#include <mirheo/core/utils/cuda_variant.h>

namespace mirheo
{

using CudaVarPtr = cuda_variant::variant<
#define MAKE_WRAPPER(a) a*
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

} // namespace mirheo
