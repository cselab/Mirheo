// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

// Prevent nvcc's invalid narrowing conversion warnings.
namespace mpark {
namespace detail {

template <typename From, typename To>
struct is_non_narrowing_convertible;

#define MIRHEO_CONVERTIBLE_(a, b)                \
    template <>                                  \
    struct is_non_narrowing_convertible<a, b> {  \
        static constexpr bool value = false;     \
    }

#define MIRHEO_CONVERTIBLE(a, b)  \
    MIRHEO_CONVERTIBLE_(a, b);    \
    MIRHEO_CONVERTIBLE_(a&, b)    \

MIRHEO_CONVERTIBLE(float, long long);
MIRHEO_CONVERTIBLE(double, long long);
MIRHEO_CONVERTIBLE(long long, float);
MIRHEO_CONVERTIBLE(long long, double);
MIRHEO_CONVERTIBLE(float, int);
MIRHEO_CONVERTIBLE(int, float);

#undef MIRHEO_CONVERTIBLE
#undef MIRHEO_CONVERTIBLE_

} // namespace detail
} // namespace mpark

#include <extern/variant/include/mpark/variant.hpp>
