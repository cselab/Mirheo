#pragma once

#include <extern/pybind11/include/pybind11/pybind11.h>

namespace pybind11
{
    template <typename type_, typename... options>
    using nodelete_class = class_< type_, std::unique_ptr<type_, nodelete>, options... >;
}
