#include "bindings.h"
#include "class_wrapper.h"

#include <pybind11/stl.h>
#include <vector_types.h>

#include <string>

using namespace pybind11::literals;

void exportVectorTypes(py::module& m)
{
    py::class_<float3>(m, "float3")
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 3)
                throw std::runtime_error("Should have length 3.");
            return float3{t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>()};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 3)
                throw std::runtime_error("Should have length 3.");
            return float3{t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>()};
        }))
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z)
        .def("__getitem__", [](const float3 &v, size_t i)
        {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            if (i == 2) return v.z;
            throw py::index_error();
            return 0.f;
        });

    py::implicitly_convertible<py::tuple, float3>();
    py::implicitly_convertible<py::list, float3>();

    py::class_<int3>(m, "int3")
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 3)
                throw std::runtime_error("Should have length 3.");
            return int3{t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>()};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 3)
                throw std::runtime_error("Should have length 3.");
            return int3{t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>()};
        }));

    py::implicitly_convertible<py::tuple, int3>();
    py::implicitly_convertible<py::list, int3>();
}
