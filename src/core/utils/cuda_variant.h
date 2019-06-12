#pragma once

#include "cpu_gpu_defines.h"

#include <extern/cuda_variant/variant/variant.h>

#include <type_traits>

namespace variant
{

template <typename T, typename VarType>
__HD__ inline bool holds_alternative(VarType var)
{
    return apply_visitor([](auto entry)
    {
        return std::is_same<T, decltype(var)>::value;
    }, var);
}

} // namespace variant

namespace cuda_variant = variant;
