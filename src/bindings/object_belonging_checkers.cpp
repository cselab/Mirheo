#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/object_belonging/interface.h>
#include <core/object_belonging/ellipsoid_belonging.h>
#include <core/object_belonging/mesh_belonging.h>

#include "nodelete.h"


namespace py = pybind11;
using namespace pybind11::literals;

void exportObjectBelongingCheckers(py::module& m)
{
    // Initial Conditions
    py::nodelete_class<ObjectBelongingChecker> pycheck(m, "BelongingChecker", R"(
        Base class for checking if particles belong to objects
    )");

    py::nodelete_class<MeshBelongingChecker>(m, "Mesh", pycheck)
        .def(py::init<std::string>(),
             "name"_a, R"(
            This checker will use the triangular mesh associated with objects to detect *inside*-*outside* status.
   
            .. note:
                Checking if particles are inside or outside the mesh is a computationally expensive task,
                so it's best to perform checks at most every 1'000 - 10'000 time-steps.
            
            Args:
                name: name of the checker
        )");
        
    py::nodelete_class<EllipsoidBelongingChecker>(m, "Ellipsoid", pycheck)
        .def(py::init<std::string>(),
             "name"_a, R"(
            This checker will use the analytical representation of the ellipsoid to detect *inside*-*outside* status.
            
            Args:
                name: name of the checker
                
            )");
}

