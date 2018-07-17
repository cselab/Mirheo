#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/integrators/factory.h>

#include "nodelete.h"

namespace py = pybind11;
using namespace pybind11::literals;

void exportIntegrators(py::module& m)
{
    // Initial Conditions
    py::nodelete_class<Integrator> pyint(m, "Integrator", R"(
        Base integration class
    )");

    py::nodelete_class<IntegratorConstOmega>(m, "Rotate", pyint)
        .def(py::init(&IntegratorFactory::createConstOmega),
             "name"_a, "dt"_a, "center"_a, "omega"_a, R"(
                Rotate particles around the specified point in space with a constant angular velocity :math:`\mathbf{\Omega}`
                
                Args:
                    name: name of the integrator
                    dt:   integration time-step
                    center: point around which to rotate
                    omega: angular velocity :math:`\mathbf{\Omega}`
            )");
        
    py::nodelete_class<IntegratorOscillate>(m, "Oscillate", pyint)
        .def(py::init(&IntegratorFactory::createOscillating),
             "name"_a, "dt"_a, "velocity"_a, "period"_a, R"(
                Move particles with the periodically changing velocity
                :math:`\mathbf{u}(t) = \cos(2 \pi \, t / T) \mathbf{u}_0`
                
                Args:
                    name: name of the integrator
                    dt:   integration time-step
                    velocity: :math:`\mathbf{u}_0`
                    period: oscillation period :math:`T`
            )");
        
    py::nodelete_class<IntegratorVVRigid>(m, "RigidVelocityVerlet", pyint)
        .def(py::init(&IntegratorFactory::createRigidVV),
             "name"_a, "dt"_a, R"(
                Integrate the position and rotation (in terms of quaternions) of the rigid bodies as per Velocity-Verlet scheme.
                Can only applied to :class:`RigidObjectVector` or :class:`RigidEllipsoidObjectVector`.
                
                Args:
                    name: name of the integrator
                    dt:   integration time-step
            )");
        
    py::nodelete_class<IntegratorTranslate>(m, "Translate", pyint)
        .def(py::init(&IntegratorFactory::createTranslate),
             "name"_a, "dt"_a, "velocity"_a, R"(
                Translate particles with a constant velocity :math:`\mathbf{u}` regardless forces acting on them.
                
                Args:
                    name: name of the integrator
                    dt:   integration time-step
                    velocity: translational velocity :math:`\mathbf{\Omega}`
            )");
        
    py::nodelete_class<IntegratorVV<Forcing_None>>
        (m, "VelocityVerlet", pyint)
        .def(py::init(&IntegratorFactory::createVV),
             "name"_a, "dt"_a, R"(
                Classical Velocity-Verlet integrator with fused steps for coordinates and velocities.
                The velocities are shifted with respect to the coordinates by one half of the time-step

                .. math::

                    \mathbf{a}^{n} &= \frac{1}{m} \mathbf{F}(\mathbf{x}^{n}, \mathbf{v}^{n-1/2}) \\
                    \mathbf{v}^{n+1/2} &= \mathbf{v}^{n-1/2} + \mathbf{a}^n \Delta t \\
                    \mathbf{x}^{n+1} &= \mathbf{x}^{n} + \mathbf{v}^{n+1/2} \Delta t 

                where bold symbol means a vector, :math:`m` is a particle mass, and superscripts denote the time: :math:`\mathbf{x}^{k} = \mathbf{x}(k \, \Delta t)`

                Args:
                    name: name of the integrator
                    dt:   integration time-step
            )");
        
    py::nodelete_class<IntegratorVV<Forcing_ConstDP>>
        (m, "VelocityVerlet_withConstForce", pyint)
        .def(py::init(&IntegratorFactory::createVV_constDP),
             "name"_a, "dt"_a, "force"_a, R"(
                Same as regular :class:`VelocityVerlet`, but the forces on all the particles are modified with the constant pressure term:
   
                .. math::

                    \mathbf{a}^{n} &= \frac{1}{m} \left( \mathbf{F}(\mathbf{x}^{n}, \mathbf{v}^{n-1/2}) + \mathbf{F}_{extra} \right) \\
                    

                Args:
                    name: name of the integrator
                    dt:   integration time-step
                    force: :math:`\mathbf{F}_{extra}`
            )");
        
    py::nodelete_class<IntegratorVV<Forcing_PeriodicPoiseuille>>
        (m, "VelocityVerlet_withPeriodicForce", pyint)
        .def(py::init(&IntegratorFactory::createVV_PeriodicPoiseuille),
             "name"_a, "dt"_a, "force"_a, "direction"_a, R"(
                Same as regular Velocity-Verlet, but the forces on all the particles are modified with periodic Poiseuille term.
                This means that all the particles in half domain along certain axis (Ox, Oy or Oz) are pushed with force
                :math:`F_{Poiseuille}` parallel to Oy, Oz or Ox correspondingly, and the particles in another half of the domain are pushed in the same direction
                with force :math:`-F_{Poiseuille}`    
   
                
                Args:
                    name: name of the integrator
                    dt:   integration time-step
                    force: orce magnitude, :math:`F_{Poiseuille}`
                    direction: Valid values: "x", "y", "z". Defines the direction of the pushing force
            )");
}

