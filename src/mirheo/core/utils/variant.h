// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

// Prevent nvcc's invalid narrowing conversion warnings.
namespace mpark {
namespace detail {

template <typename From, typename To>
struct is_non_narrowing_convertible;

template <>
struct is_non_narrowing_convertible<double, long long> {
    static constexpr bool value = false;
};

template <>
struct is_non_narrowing_convertible<double&, long long> {
    static constexpr bool value = false;
};

template <>
struct is_non_narrowing_convertible<long long, double> {
    static constexpr bool value = false;
};

template <>
struct is_non_narrowing_convertible<long long&, double> {
    static constexpr bool value = false;
};

} // namespace detail
} // namespace mpark

#include <extern/variant/include/mpark/variant.hpp>
