// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <pybind11/pybind11.h>
#include <memory>

namespace pybind11
{
    template <typename type_, typename... options>
    using handlers_class = class_< type_, std::shared_ptr<type_>, options... >;
}
