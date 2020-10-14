// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "bindings.h"
#include "class_wrapper.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/strprintf.h>

#include <pybind11/stl.h>
#include <vector_types.h>

#include <string>

namespace mirheo
{

using namespace pybind11::literals;

void exportVectorTypes(py::module& m)
{
    // NOTE: We use std::vector<T> in constructors below because it
    // seems to cover most implicit conversions (from tuples, lists,
    // numpy arrays). std::array<T, N> does not work so well.
    py::class_<real2>(m, "real2")
        .def(py::init([](real x, real y) {
            return real2{x, y};
        }))
        .def(py::init([](const std::vector<real> &v)
        {
            if (v.size() != 2)
                throw std::invalid_argument("Expected 2 elements.");
            return real2{v[0], v[1]};
        }))
        .def_readwrite("x", &real2::x)
        .def_readwrite("y", &real2::y)
        .def("__getitem__", [](const real2 &v, size_t i)
        {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            throw py::index_error();
        })
        .def("__str__", [](const real2 &v) {
            return "({}, {})"_s.format(v.x, v.y);
        })
        .def("__repr__", [](const real2 &v) {
            return "real2({}, {})"_s.format(v.x, v.y);
        });

    py::implicitly_convertible<py::iterable, real2>();


    py::class_<real3>(m, "real3")
        .def(py::init([](real x, real y, real z) {
            return real3{x, y, z};
        }))
        .def(py::init([](const std::vector<real> &v)
        {
            if (v.size() != 3)
                throw std::invalid_argument("Expected 3 elements.");
            return real3{v[0], v[1], v[2]};
        }))
        .def_readwrite("x", &real3::x)
        .def_readwrite("y", &real3::y)
        .def_readwrite("z", &real3::z)
        .def("__getitem__", [](const real3 &v, size_t i)
        {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            if (i == 2) return v.z;
            throw py::index_error();
        })
        .def("__str__", [](const real3 &v) {
            return "({}, {}, {})"_s.format(v.x, v.y, v.z);
        })
        .def("__repr__", [](const real3 &v) {
            return "real3({}, {}, {})"_s.format(v.x, v.y, v.z);
        });

    py::implicitly_convertible<py::iterable, real3>();


    py::class_<real4>(m, "real4")
        .def(py::init([](real x, real y, real z, real w) {
            return real4{x, y, z, w};
        }))
        .def(py::init([](const std::vector<real> &v)
        {
            if (v.size() != 4)
                throw std::invalid_argument("Expected 4 elements.");
            return real4{v[0], v[1], v[2], v[3]};
        }))
        .def_readwrite("x", &real4::x)
        .def_readwrite("y", &real4::y)
        .def_readwrite("z", &real4::z)
        .def_readwrite("w", &real4::w)
        .def("__getitem__", [](const real4 &v, size_t i)
        {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            if (i == 2) return v.z;
            if (i == 3) return v.z;
            throw py::index_error();
        })
        .def("__str__", [](const real4 &v) {
            return "({}, {}, {}, {})"_s.format(v.x, v.y, v.z, v.w);
        })
        .def("__repr__", [](const real4 &v) {
            return "real4({}, {}, {}, {})"_s.format(v.x, v.y, v.z, v.w);
        });

    py::implicitly_convertible<py::iterable, real4>();


    py::class_<int3>(m, "int3")
        .def(py::init([](int x, int y, int z) {
            return int3{x, y, z};
        }))
        .def(py::init([](const std::vector<int> &v)
        {
            if (v.size() != 3)
                throw std::invalid_argument("Expected 3 elements.");
            return int3{v[0], v[1], v[2]};
        }))
        .def_readwrite("x", &int3::x)
        .def_readwrite("y", &int3::y)
        .def_readwrite("z", &int3::z)
        .def("__str__", [](const int3 &v) {
            return "({}, {}, {})"_s.format(v.x, v.y, v.z);
        })
        .def("__repr__", [](const int3 &v) {
            return "int3({}, {}, {})"_s.format(v.x, v.y, v.z);
        });

    py::implicitly_convertible<py::iterable, int3>();


    py::class_<ComQ>(m, "ComQ")
        .def(py::init([](real x, real y, real z, real qx, real qy, real qz, real qw) {
            return ComQ{real3{x, y, z}, real4{qx, qy, qz, qw}};
        }))
        .def(py::init([](real3 r, real4 q) {
            return ComQ{r, q};
        }))
        .def(py::init([](const std::vector<real> &v)
        {
            if (v.size() != 7)
                throw std::invalid_argument("Expected 7 elements.");
            const real3 com {v[0], v[1], v[2]};
            const real4 Q {v[3], v[4], v[5], v[6]};
            return ComQ{com, Q};
        }))
        .def("__repr__", [](const ComQ &cq) {
            return "ComQ({}, {}, {}; {}, {}, {}, {})"_s.format(
                    cq.r.x, cq.r.y, cq.r.z, cq.q.x, cq.q.y, cq.q.z, cq.q.w);
        });

    py::implicitly_convertible<py::iterable, ComQ>();
}

} // namespace mirheo
