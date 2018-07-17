#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/bouncers/interface.h>
#include <core/bouncers/from_ellipsoid.h>
#include <core/bouncers/from_mesh.h>

#include "nodelete.h"


namespace py = pybind11;
using namespace pybind11::literals;

void exportBouncers(py::module& m)
{
    // Initial Conditions
    py::nodelete_class<Bouncer> pybounce(m, "Bouncer", R"(
        Base class for bouncing particles off the objects
        
        Bouncers prevent liquid particles crossing boundaries of objects (maintaining no-through boundary conditions).
        The idea of the bouncers is to move the particles that crossed the object boundary after integration step back to the correct side.
        Particles are moved such that they appear very close (about :math:`10^{-4}` units away from the boundary).
        Assuming that the objects never come too close to each other or the walls,
        that approach ensures that recovered particles will not penetrate into a different object or wall.
        In practice maintaining separation of at least :math:`10^{-3}` units between walls and objects is sufficient.
        Note that particle velocities are also altered, which means that objects experience extra force from the collisions.
        The force from a collision is applied at the beginning of the following time-step.
    )");

    py::nodelete_class<BounceFromMesh>(m, "Mesh", pybounce)
        .def(py::init<std::string, float>(),
             "name"_a, "kbt"_a=0.5, R"(
            This bouncer will use the triangular mesh associated with objects to detect boundary crossings.
            Therefore it can only be created for Membrane and Rigid Object types of object vectors.
            Due to numerical precision, about :math:`1` of :math:`10^5 - 10^6` mesh crossings will not be detected, therefore it is advised to use that bouncer in
            conjunction with correction option provided by the Object Belonging Checker, see :ref:`user-belongers`.
            
            .. note:
                In order to prevent numerical instabilities in case of light membrane particles,
                the new velocity of the bounced particles will be a random vector drawn from the Maxwell distibution of given temperature
                and added to the velocity of the mesh triangle at the collision point.
            
            Args:
                name: name of the bouncer
                kbt:  Maxwell distribution temperature defining post-collision velocity
        )");
        
    py::nodelete_class<BounceFromRigidEllipsoid>(m, "Ellipsoid", pybounce)
        .def(py::init<std::string>(),
             "name"_a, R"(
            This bouncer will use the analytical ellipsoid representation to perform the bounce.
            No additional correction from the Object Belonging Checker is usually required.
            The velocity of the particles bounced from the ellipsoid is reversed with respect to the boundary velocity at the contact point.
            
            Args:
                name: name of the checker
                
            )");
}

