#include "bindings.h"
#include "class_wrapper.h"

#include <core/datatypes.h>

#include <pybind11/stl.h>
#include <vector_types.h>

#include <string>

using namespace pybind11::literals;

void exportVectorTypes(py::module& m)
{
    py::class_<real2>(m, "real2")
        .def(py::init([](real x, real y) {
            return real2{x, y};
        }))
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 2)
                throw std::runtime_error("Should have length 2.");
            return real2{t[0].cast<real>(), t[1].cast<real>()};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 2)
                throw std::runtime_error("Should have length 2.");
            return real2{t[0].cast<real>(), t[1].cast<real>()};
        }))
        .def_readwrite("x", &real2::x)
        .def_readwrite("y", &real2::y)
        .def("__getitem__", [](const real2 &v, size_t i)
        {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            throw py::index_error();
            return 0._r;
        })
        .def("__str__", [](const real2 &v) {
            return "({}, {})"_s.format(v.x, v.y);
        })
        .def("__repr__", [](const real2 &v) {
            return "real2({}, {})"_s.format(v.x, v.y);
        });

    py::implicitly_convertible<py::tuple, real2>();
    py::implicitly_convertible<py::list,  real2>();

    
    py::class_<real3>(m, "real3")
        .def(py::init([](real x, real y, real z) {
            return real3{x, y, z};
        }))
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 3)
                throw std::runtime_error("Should have length 3.");
            return real3{t[0].cast<real>(), t[1].cast<real>(), t[2].cast<real>()};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 3)
                throw std::runtime_error("Should have length 3.");
            return real3{t[0].cast<real>(), t[1].cast<real>(), t[2].cast<real>()};
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
            return 0._r;
        })
        .def("__str__", [](const real3 &v) {
            return "({}, {}, {})"_s.format(v.x, v.y, v.z);
        })
        .def("__repr__", [](const real3 &v) {
            return "real3({}, {}, {})"_s.format(v.x, v.y, v.z);
        });

    py::implicitly_convertible<py::tuple, real3>();
    py::implicitly_convertible<py::list,  real3>();


    py::class_<real4>(m, "real4")
        .def(py::init([](real x, real y, real z, real w) {
            return real4{x, y, z, w};
        }))
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 4)
                throw std::runtime_error("Should have length 4.");
            return real4{t[0].cast<real>(), t[1].cast<real>(), t[2].cast<real>(), t[3].cast<real>()};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 4)
                throw std::runtime_error("Should have length 4.");
            return real4{t[0].cast<real>(), t[1].cast<real>(), t[2].cast<real>(), t[3].cast<real>()};
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
            return 0._r;
        })
        .def("__str__", [](const real4 &v) {
            return "({}, {}, {}, {})"_s.format(v.x, v.y, v.z, v.w);
        })
        .def("__repr__", [](const real4 &v) {
            return "real4({}, {}, {}, {})"_s.format(v.x, v.y, v.z, v.w);
        });

    py::implicitly_convertible<py::tuple, real4>();
    py::implicitly_convertible<py::list,  real4>();

    
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
        .def(py::init([](real x, real y, real z, real qx, real qy, real qz, real qw) {
            return ComQ{real3{x, y, z}, real4{qx, qy, qz, qw}};
        }))
        .def(py::init([](real3 r, real4 q) {
            return ComQ{r, q};
        }))
        .def(py::init([](py::tuple t)
        {
            if (py::len(t) != 7)
                throw std::runtime_error("Should have length 7.");

            const real3 com {t[0].cast<real>(), t[1].cast<real>(), t[2].cast<real>()};
            const real4 Q {t[3].cast<real>(), t[4].cast<real>(), t[5].cast<real>(), t[6].cast<real>()};
            return ComQ{com, Q};
        }))
        .def(py::init([](py::list t)
        {
            if (py::len(t) != 7)
                throw std::runtime_error("Should have length 7.");

            const real3 com {t[0].cast<real>(), t[1].cast<real>(), t[2].cast<real>()};
            const real4 Q {t[3].cast<real>(), t[4].cast<real>(), t[5].cast<real>(), t[6].cast<real>()};
            return ComQ{com, Q};
        }))
        .def("__repr__", [](const ComQ &cq) {
            return "ComQ({}, {}, {}; {}, {}, {}, {})"_s.format(
                    cq.r.x, cq.r.y, cq.r.z, cq.q.x, cq.q.y, cq.q.z, cq.q.w);
        });

    py::implicitly_convertible<py::tuple, ComQ>();
    py::implicitly_convertible<py::list,  ComQ>();
}
