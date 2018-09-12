#include <pybind11/pybind11.h>

#include <core/bouncers/interface.h>
#include <core/bouncers/from_ellipsoid.h>
#include <core/bouncers/from_mesh.h>

#include "class_wrapper.h"


namespace py = pybind11;
using namespace pybind11::literals;

void exportBouncers(py::module& m)
{
    py::handlers_class<Bouncer> pybounce(m, "Bouncer", R"(
        Base class for bouncing particles off the objects
    )");

    
    py::handlers_class<BounceFromMesh>(m, "Mesh", pybounce, R"(
        This bouncer will use the triangular mesh associated with objects to detect boundary crossings.
        Therefore it can only be created for Membrane and Rigid Object types of object vectors.
        Due to numerical precision, about :math:`1` of :math:`10^5 - 10^6` mesh crossings will not be detected, therefore it is advised to use that bouncer in
        conjunction with correction option provided by the Object Belonging Checker, see :ref:`user-belongers`.
        
        .. note:
            In order to prevent numerical instabilities in case of light membrane particles,
            the new velocity of the bounced particles will be a random vector drawn from the Maxwell distibution of given temperature
            and added to the velocity of the mesh triangle at the collision point.
    )")
        .def(py::init<std::string, float>(), "name"_a, "kbt"_a=0.5, R"(
            Args:
                name: name of the bouncer
                kbt:  Maxwell distribution temperature defining post-collision velocity
        )");
        
    py::handlers_class<BounceFromRigidEllipsoid>(m, "Ellipsoid", pybounce, R"(
        This bouncer will use the analytical ellipsoid representation to perform the bounce.
        No additional correction from the Object Belonging Checker is usually required.
        The velocity of the particles bounced from the ellipsoid is reversed with respect to the boundary velocity at the contact point.
    )")
        .def(py::init<std::string>(),
             "name"_a, R"(
            Args:
                name: name of the checker
            
        )");
}

