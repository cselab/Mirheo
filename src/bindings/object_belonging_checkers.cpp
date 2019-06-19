#include "bindings.h"
#include "class_wrapper.h"

#include <core/analytical_shapes/api.h>
#include <core/object_belonging/shape_belonging.h>
#include <core/object_belonging/interface.h>
#include <core/object_belonging/mesh_belonging.h>
#include <core/object_belonging/rod_belonging.h>

using namespace pybind11::literals;

void exportObjectBelongingCheckers(py::module& m)
{
    py::handlers_class<ObjectBelongingChecker> pycheck(m, "BelongingChecker", R"(
        Base class for checking if particles belong to objects
    )");

    py::handlers_class<MeshBelongingChecker>(m, "Mesh", pycheck, R"(
        This checker will use the triangular mesh associated with objects to detect *inside*-*outside* status.
   
        .. note:
            Checking if particles are inside or outside the mesh is a computationally expensive task,
            so it's best to perform checks at most every 1'000 - 10'000 time-steps.
    )")
        .def(py::init<const YmrState*, std::string>(),
             "state"_a, "name"_a, R"(
            Args:
                name: name of the checker
        )");
        
    py::handlers_class<ShapeBelongingChecker<Capsule>>(m, "Capsule", pycheck, R"(
        This checker will use the analytical representation of the capsule to detect *inside*-*outside* status.
    )")
        .def(py::init<const YmrState*, std::string>(),
             "state"_a, "name"_a, R"(
            Args:
                name: name of the checker
            )");

    py::handlers_class<ShapeBelongingChecker<Cylinder>>(m, "Cylinder", pycheck, R"(
        This checker will use the analytical representation of the cylinder to detect *inside*-*outside* status.
    )")
        .def(py::init<const YmrState*, std::string>(),
             "state"_a, "name"_a, R"(
            Args:
                name: name of the checker
            )");

    py::handlers_class<ShapeBelongingChecker<Ellipsoid>>(m, "Ellipsoid", pycheck, R"(
        This checker will use the analytical representation of the ellipsoid to detect *inside*-*outside* status.
    )")
        .def(py::init<const YmrState*, std::string>(),
             "state"_a, "name"_a, R"(
            Args:
                name: name of the checker
            )");

    py::handlers_class<RodBelongingChecker>(m, "Rod", pycheck, R"(
        This checker will detect *inside*-*outside* status with respect to every segment of the rod, enlarged by a given radius.
    )")
        .def(py::init<const YmrState*, std::string, float>(),
             "state"_a, "name"_a, "radius"_a, R"(
            Args:
                name: name of the checker
                radius: radius of the rod
            )");
}

