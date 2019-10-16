#include "bindings.h"
#include "class_wrapper.h"

#include <core/datatypes.h>

#include <pybind11/stl.h>
#include <vector_types.h>

#include <string>

using namespace pybind11::literals;

void exportVectorTypes(py::module& m)
{
    py::class_<float2>(m, "float2")
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 2)
                throw std::runtime_error("Should have length 2.");
            return float2{t[0].cast<float>(), t[1].cast<float>()};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 2)
                throw std::runtime_error("Should have length 2.");
            return float2{t[0].cast<float>(), t[1].cast<float>()};
        }))
        .def_readwrite("x", &float2::x)
        .def_readwrite("y", &float2::y)
        .def("__getitem__", [](const float2 &v, size_t i)
        {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            throw py::index_error();
            return 0.f;
        });

    py::implicitly_convertible<py::tuple, float2>();
    py::implicitly_convertible<py::list,  float2>();

    
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
    py::implicitly_convertible<py::list,  float3>();


    py::class_<float4>(m, "float4")
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 4)
                throw std::runtime_error("Should have length 4.");
            return float4{t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>()};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 4)
                throw std::runtime_error("Should have length 4.");
            return float4{t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>()};
        }))
        .def_readwrite("x", &float4::x)
        .def_readwrite("y", &float4::y)
        .def_readwrite("z", &float4::z)
        .def_readwrite("w", &float4::w)
        .def("__getitem__", [](const float4 &v, size_t i)
        {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            if (i == 2) return v.z;
            if (i == 3) return v.z;
            throw py::index_error();
            return 0.f;
        });

    py::implicitly_convertible<py::tuple, float4>();
    py::implicitly_convertible<py::list,  float4>();

    
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
    py::implicitly_convertible<py::list,  int3>();



    py::class_<ComQ>(m, "ComQ")
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 7)
                throw std::runtime_error("Should have length 7.");

            const float3 com {t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>()};
            const float4 Q {t[3].cast<float>(), t[4].cast<float>(), t[5].cast<float>(), t[6].cast<float>()};
            return ComQ{com, Q};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 7)
                throw std::runtime_error("Should have length 7.");

            const float3 com {t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>()};
            const float4 Q {t[3].cast<float>(), t[4].cast<float>(), t[5].cast<float>(), t[6].cast<float>()};
            return ComQ{com, Q};
        }));

    py::implicitly_convertible<py::tuple, ComQ>();
    py::implicitly_convertible<py::list,  ComQ>();
}
