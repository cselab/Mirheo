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
        .def(py::init([](float x, float y) {
            return float2{x, y};
        }))
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
        })
        .def("__str__", [](const float2 &v) {
            return "({}, {})"_s.format(v.x, v.y);
        })
        .def("__repr__", [](const float2 &v) {
            return "float2({}, {})"_s.format(v.x, v.y);
        });

    py::implicitly_convertible<py::tuple, float2>();
    py::implicitly_convertible<py::list,  float2>();

    
    py::class_<float3>(m, "float3")
        .def(py::init([](float x, float y, float z) {
            return float3{x, y, z};
        }))
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
        })
        .def("__str__", [](const float3 &v) {
            return "({}, {}, {})"_s.format(v.x, v.y, v.z);
        })
        .def("__repr__", [](const float3 &v) {
            return "float3({}, {}, {})"_s.format(v.x, v.y, v.z);
        });

    py::implicitly_convertible<py::tuple, float3>();
    py::implicitly_convertible<py::list,  float3>();


    py::class_<float4>(m, "float4")
        .def(py::init([](float x, float y, float z, float w) {
            return float4{x, y, z, w};
        }))
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
        })
        .def("__str__", [](const float4 &v) {
            return "({}, {}, {}, {})"_s.format(v.x, v.y, v.z, v.w);
        })
        .def("__repr__", [](const float4 &v) {
            return "float4({}, {}, {}, {})"_s.format(v.x, v.y, v.z, v.w);
        });

    py::implicitly_convertible<py::tuple, float4>();
    py::implicitly_convertible<py::list,  float4>();

    
    py::class_<int3>(m, "int3")
        .def(py::init([](int x, int y, int z) {
            return int3{x, y, z};
        }))
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
        }))
        .def("__str__", [](const int3 &v) {
            return "({}, {}, {})"_s.format(v.x, v.y, v.z);
        })
        .def("__repr__", [](const int3 &v) {
            return "int3({}, {}, {})"_s.format(v.x, v.y, v.z);
        });

    py::implicitly_convertible<py::tuple, int3>();
    py::implicitly_convertible<py::list,  int3>();



    py::class_<ComQ>(m, "ComQ")
        .def(py::init([](float x, float y, float z, float qx, float qy, float qz, float qw) {
            return ComQ{float3{x, y, z}, float4{qx, qy, qz, qw}};
        }))
        .def(py::init([](float3 r, float4 q) {
            return ComQ{r, q};
        }))
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
        }))
        .def("__repr__", [](const ComQ &cq) {
            return "ComQ({}, {}, {}; {}, {}, {}, {})"_s.format(
                    cq.r.x, cq.r.y, cq.r.z, cq.q.x, cq.q.y, cq.q.z, cq.q.w);
        });

    py::implicitly_convertible<py::tuple, ComQ>();
    py::implicitly_convertible<py::list,  ComQ>();
}
