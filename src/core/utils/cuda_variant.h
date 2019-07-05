#pragma once

#include "cpu_gpu_defines.h"

#include <extern/cuda_variant/variant/variant.h>

#include <type_traits>

namespace variant
{

template <typename T, typename Variant>
__HD__ inline bool holds_alternative(const Variant& var)
{
    return apply_visitor([](auto entry)
    {
        return std::is_same<T, decltype(entry)>::value;
    }, var);
}

namespace details
{
template <typename T>
struct Getter {
    __HD__ inline T operator()(T in) const {return in;}

    template <typename X>
    __HD__ inline T operator()(X) const {return *static_cast<T*>(nullptr);}
};
} // namespace details

template <typename T, typename Variant>
__HD__ inline T get_alternative(const Variant& var)
{
    return apply_visitor(details::Getter<T>(), var);
}

} // namespace variant

namespace cuda_variant = variant;
