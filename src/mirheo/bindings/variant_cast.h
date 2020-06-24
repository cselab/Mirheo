// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <extern/variant/include/mpark/variant.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// provide cast for `mpark::variant` as explained in
// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers

namespace pybind11 {
namespace detail {

template <typename... Ts>
struct type_caster<mpark::variant<Ts...>> : variant_caster<mpark::variant<Ts...>> {};

template <>
struct visit_helper<mpark::variant> {
    template <typename... Args>
    static auto call(Args &&...args) -> decltype(mpark::visit(args...)) {
        return mpark::visit(args...);
    }
};

}} // namespace pybind11::detail
