#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/mesh.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/membrane_vector.h>

#include <core/utils/pytypes.h>

#include "class_wrapper.h"

namespace py = pybind11;
using namespace pybind11::literals;

void exportParticleVectors(py::module& m)
{
    py::handlers_class<ParticleVector> pypv(m, "ParticleVector", R"(
        Basic particle vector, consists of identical disconnected particles.
    )");
    
    pypv.def(py::init<std::string, float>(), py::return_value_policy::move,"name"_a, "mass"_a, R"(
            Args:
                name: name of the created PV 
                mass: mass of a single particle
        )")
        //
        .def("get_indices", &ParticleVector::getIndices_vector, R"(
            Returns:
                A list of unique integer particle identifiers
        )")
        .def("getCoordinates", &ParticleVector::getCoordinates_vector, R"(
            Returns: 
                A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        )")
        .def("getVelocities",  &ParticleVector::getVelocities_vector, R"(
            Returns: 
                A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        )")
        .def("getForces",      &ParticleVector::getForces_vector, R"(
            Returns: 
                A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        )")
        //
        .def("setCoordinates", &ParticleVector::setCoordinates_vector, "coordinates"_a, R"(
            Args:
                coordinates: A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        )")
        .def("setVelocities",  &ParticleVector::setVelocities_vector, "velocities"_a, R"(
            Args:
                velocities: A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        )")
        .def("setForces",      &ParticleVector::setForces_vector, "forces"_a, R"(
            Args:
                forces: A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        )");

    py::handlers_class<Mesh> pymesh(m, "Mesh", R"(
        Internally used class for desctibing a simple triangular mesh
    )");

    pymesh.def(py::init<std::string>(), "off_filename"_a, R"(
        Create a mesh by reading the OFF file
        Args:
            off_filename: path of the OFF file
    )")
        .def(py::init<const PyTypes::VectorOfFloat3&, const PyTypes::VectorOfInt3&>(), "vertices"_a, "faces"_a, R"(
        Create a mesh by giving coordinates and connectivity
        Args:
            vertices: vertex coordinates
            faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices
    )")
        .def("getVertices", &Mesh::getVertices, R"(
        returns the vertex coordinates of the mesh.
    )")
        .def("getTriangles", &Mesh::getTriangles, R"(
        returns the vertex indices for each triangle of the mesh.
    )");

    py::handlers_class<MembraneMesh>(m, "MembraneMesh", pymesh, R"(
        Internally used class for desctibing a triangular mesh that can be used with the Membrane Interactions.
        In contrast with the simple :any:`Mesh`, this class precomputes some required quantities on the mesh
    )")
        .def(py::init<std::string>(), "off_filename"_a, R"(
            Create a mesh by reading the OFF file
            Args:
                off_filename: path of the OFF file
        )")
        .def(py::init<const PyTypes::VectorOfFloat3&, const PyTypes::VectorOfInt3&>(), "vertices"_a, "faces"_a, R"(
        Create a mesh by giving coordinates and connectivity
        Args:
            vertices: vertex coordinates
            faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices
    )");
        
    py::handlers_class<ObjectVector> pyov(m, "ObjectVector", pypv, R"(
        Basic Object Vector
    )"); 
        
    py::handlers_class<MembraneVector> (m, "MembraneVector", pyov, R"(
        Membrane is an Object Vector representing cell membranes.
        It must have a triangular mesh associated with it such that each particle is mapped directly onto single mesh vertex.
    )")
        .def(py::init<std::string, float, std::shared_ptr<MembraneMesh>>(),
             "name"_a, "mass"_a, "mesh"_a, R"(
            Args:
                name: name of the created PV 
                mass: mass of a single particle
                mesh: :any:`MembraneMesh` object                
        )");
        
    py::handlers_class<RigidObjectVector> (m, "RigidObjectVector", pyov, R"(
        Rigid Object is an Object Vector representing objects that move as rigid bodies, with no relative displacement against each other in an object.
        It must have a triangular mesh associated with it that defines the shape of the object.
    )")
        .def(py::init<std::string, float, PyTypes::float3, int, std::shared_ptr<Mesh>>(),
             "name"_a, "mass"_a, "inertia"_a, "object_size"_a, "mesh"_a, R"( 
                Args:
                    name: name of the created PV 
                    mass: mass of a single particle
                    inertia: moment of inertia of the body in its principal axes. The principal axes of the mesh are assumed to be aligned with the default global *OXYZ* axes
                    object_size: number of particles per membrane, must be the same as the number of vertices of the mesh
                    mesh: :any:`MembraneMesh` object         
        )");
        
    py::handlers_class<RigidEllipsoidObjectVector> (m, "RigidEllipsoidVector", pyov, R"(
        Rigid Ellipsoid is the same as the Rigid Object except that it can only represent ellipsoidal shapes.
        The advantage is that it doesn't need mesh and moment of inertia define, as those can be computed analytically.
    )")
        .def(py::init<std::string, float, int, PyTypes::float3>(),
             "name"_a, "mass"_a, "object_size"_a, "semi_axes"_a, R"(
                Args:
                    name: name of the created PV 
                    mass: mass of a single particle
                    object_size: number of particles per membrane, must be the same as the number of vertices of the mesh
                    semi_axes: ellipsoid principal semi-axes
        )")
        .def(py::init<std::string, float, int, PyTypes::float3, std::shared_ptr<Mesh>>(),
             "name"_a, "mass"_a, "object_size"_a, "semi_axes"_a, "mesh"_a, R"(
                Args:
                    name: name of the created PV 
                    mass: mass of a single particle
                    object_size: number of particles per membrane, must be the same as the number of vertices of the mesh
                    semi_axes: ellipsoid principal semi-axes
                    mesh: mesh representing the shape of the ellipsoid. This is used for dump only.
        )");
}
