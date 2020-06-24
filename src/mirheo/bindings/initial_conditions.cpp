// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "bindings.h"
#include "class_wrapper.h"

#include <mirheo/core/initial_conditions/from_array.h>
#include <mirheo/core/initial_conditions/interface.h>
#include <mirheo/core/initial_conditions/membrane.h>
#include <mirheo/core/initial_conditions/membrane_with_type_ids.h>
#include <mirheo/core/initial_conditions/restart.h>
#include <mirheo/core/initial_conditions/rigid.h>
#include <mirheo/core/initial_conditions/rod.h>
#include <mirheo/core/initial_conditions/uniform.h>
#include <mirheo/core/initial_conditions/uniform_filtered.h>
#include <mirheo/core/initial_conditions/uniform_sphere.h>

#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace mirheo
{

using namespace pybind11::literals;

void exportInitialConditions(py::module& m)
{
    py::handlers_class<InitialConditions> pyic(m, "InitialConditions", R"(
            Base class for initial conditions
        )");


    py::handlers_class<FromArrayIC>(m, "FromArray", pyic, R"(
        Set particles according to given position and velocity arrays.
    )")
        .def(py::init<const std::vector<real3>&, const std::vector<real3>&>(),
             "pos"_a, "vel"_a, R"(
            Args:
                pos: array of positions
                vel: array of velocities
        )");

    py::handlers_class<MembraneIC> pyMembraneIC(m, "Membrane", pyic, R"(
        Can only be used with Membrane Object Vector, see :ref:`user-ic`. These IC will initialize the particles of each object
        according to the mesh associated with Membrane, and then the objects will be translated/rotated according to the provided initial conditions.
    )");

    pyMembraneIC.def(py::init<const std::vector<ComQ>&, real>(),
                     "com_q"_a, "global_scale"_a=1.0, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.
                    One entry in the list corresponds to one object created.
                    Each entry consist of 7 reals: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized
                global_scale:
                    All the membranes will be scaled by that value. Useful to implement membranes growth so that they
                    can fill the space with high volume fraction
    )");

    py::handlers_class<MembraneWithTypeIdsIC>(m, "MembraneWithTypeId", pyMembraneIC, R"(
        Same as :class:`~libmirheo.InitialConditions.Membrane` with an additional `type id` field which distinguish membranes with different properties.
        This is may be used with :class:`~libmirheo.Interactions.MembraneForces` with the corresponding filter.
    )")
        .def(py::init<const std::vector<ComQ>&, const std::vector<int>&, real>(),
             "com_q"_a, "type_ids"_a, "global_scale"_a=1.0, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.
                    One entry in the list corresponds to one object created.
                    Each entry consist of 7 reals: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized
                type_ids:
                    list of type ids. Each entry corresponds to the id of the group to which the corresponding membrane belongs.
                global_scale:
                    All the membranes will be scaled by that value. Useful to implement membranes growth so that they
                    can fill the space with high volume fraction
        )");

    py::handlers_class<RestartIC>(m, "Restart", pyic, R"(
        Read the state of the particle vector from restart files.
    )")
        .def(py::init<std::string>(),"path"_a = "restart/", R"(

            Args:
                path: folder where the restart files reside.
        )");

    py::handlers_class<RigidIC>(m, "Rigid", pyic, R"(
        Can only be used with Rigid Object Vector or Rigid Ellipsoid, see :ref:`user-ic`. These IC will initialize the particles of each object
        according to the template .xyz file and then the objects will be translated/rotated according to the provided initial conditions.

    )")
        .def(py::init<const std::vector<ComQ>&, const std::string&>(),
             "com_q"_a, "xyz_filename"_a, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.
                    One entry in the list corresponds to one object created.
                    Each entry consist of 7 reals: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized
                xyz_filename:
                    Template that describes the positions of the body particles before translation or
                    rotation is applied. Standard .xyz file format is used with first line being
                    the number of particles, second comment, third and onwards - particle coordinates.
                    The number of particles in the file must be the same as in number of particles per object
                    in the corresponding PV
        )")
        .def(py::init<const std::vector<ComQ>&, const std::vector<real3>&>(),
             "com_q"_a, "coords"_a, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.
                    One entry in the list corresponds to one object created.
                    Each entry consist of 7 reals: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized
                coords:
                    Template that describes the positions of the body particles before translation or
                    rotation is applied.
                    The number of coordinates must be the same as in number of particles per object
                    in the corresponding PV
        )")
        .def(py::init<const std::vector<ComQ>&, const std::vector<real3>&, const std::vector<real3>&>(),
             "com_q"_a, "coords"_a, "init_vels"_a, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.
                    One entry in the list corresponds to one object created.
                    Each entry consist of 7 reals: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized
                coords:
                    Template that describes the positions of the body particles before translation or
                    rotation is applied.
                    The number of coordinates must be the same as in number of particles per object
                    in the corresponding PV
                com_q:
                    List specifying initial Center-Of-Mass velocities of the bodies.
                    One entry (list of 3 reals) in the list corresponds to one object
        )");


    py::handlers_class<RodIC>(m, "Rod", pyic, R"(
        Can only be used with Rod Vector. These IC will initialize the particles of each rod
        according to the the given explicit center-line position aand torsion mapping and then
        the objects will be translated/rotated according to the provided initial conditions.

    )")
        .def(py::init<const std::vector<ComQ>&, std::function<real3(real)>, std::function<real(real)>, real, real3>(),
             "com_q"_a, "center_line"_a, "torsion"_a, "a"_a, "initial_frame"_a=RodIC::DefaultFrame, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.
                    One entry in the list corresponds to one object created.
                    Each entry consist of 7 reals: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized
                center_line:
                    explicit mapping :math:`\mathbf{r} : [0,1] \rightarrow R^3`.
                    Assume :math:`|r'(s)|` is constant for all :math:`s \in [0,1]`.
                torsion:
                    explicit mapping :math:`\tau : [0,1] \rightarrow R`.
                a:
                    width of the rod
                initial_frame:
                    Orientation of the initial frame (optional)
                    By default, will come up with any orthogonal frame to the rod at origin
        )");

    py::handlers_class<UniformIC>(m, "Uniform", pyic, R"(
        The particles will be generated with the desired number density uniformly at random in all the domain.
        These IC may be used with any Particle Vector, but only make sense for regular PV.

    )")
        .def(py::init<real>(), "number_density"_a, R"(
            Args:
                number_density: target number density
        )");

    py::handlers_class<UniformFilteredIC>(m, "UniformFiltered", pyic, R"(
        The particles will be generated with the desired number density uniformly at random in all the domain and then filtered out by the given filter.
        These IC may be used with any Particle Vector, but only make sense for regular PV.
    )")
        .def(py::init<real, std::function<bool(real3)>>(),
             "number_density"_a, "filter"_a, R"(
            Args:
                number_density: target number density
                filter: given position, returns True if the particle should be kept
        )");

    py::handlers_class<UniformSphereIC>(m, "UniformSphere", pyic, R"(
        The particles will be generated with the desired number density uniformly at random inside or outside a given sphere.
        These IC may be used with any Particle Vector, but only make sense for regular PV.

    )")
        .def(py::init<real, real3, real, bool>(),
             "number_density"_a, "center"_a, "radius"_a, "inside"_a, R"(
            Args:
                number_density: target number density
                center: center of the sphere
                radius: radius of the sphere
                inside: whether the particles should be inside or outside the sphere
        )");
}

} // namespace mirheo
