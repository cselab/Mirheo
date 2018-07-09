#include <extern/pybind11/include/pybind11/pybind11.h>
#include <extern/pybind11/include/pybind11/stl.h>

#include <core/logger.h>

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/mesh.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/membrane_vector.h>

#include <core/initial_conditions/interface.h>
#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/rigid_ic.h>
#include <core/initial_conditions/restart.h>
#include <core/initial_conditions/membrane_ic.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_udevicex, m)
{
    // Particle Vectors
    py::class_<ParticleVector> pypv(m, "ParticleVector");
    pypv.def(py::init<std::string, float>(), "name"_a, "mass"_a)
        //
        .def("getCoordinates", &ParticleVector::getCoordinates_vector)
        .def("getVelocities",  &ParticleVector::getVelocities_vector)
        .def("getForces",      &ParticleVector::getForces_vector)
        //
        .def("setCoordinates", &ParticleVector::setCoordinates_vector)
        .def("setVelocities",  &ParticleVector::setVelocities_vector)
        .def("setForces",      &ParticleVector::setForces_vector);

    py::class_<Mesh> pymesh(m, "Mesh");
    pymesh.def(py::init<std::string>(), "off_filename"_a);
    
    py::class_<MembraneMesh>(m, "MembraneMesh", pymesh)
        .def(py::init<std::string>(), "off_filename"_a);
        
    py::class_<ObjectVector> (m, "ObjectVector", pypv)
        .def(py::init<std::string, float, int>(), "name"_a, "mass"_a, "n_objects"_a=0);  
        
    py::class_<ObjectVector> (m, "RigidObjectVector", pypv)
        .def(py::init<std::string, float, int>(), "name"_a, "mass"_a, "n_objects"_a=0);  
        
    py::class_<ObjectVector> (m, "ObjectVector", pypv)
        .def(py::init<std::string, float, int>(), "name"_a, "mass"_a, "n_objects"_a=0);    
    py::class_<ObjectVector> (m, "ObjectVector", pypv)
        .def(py::init<std::string, float, int>(), "name"_a, "mass"_a, "n_objects"_a=0);    
    py::class_<ObjectVector> (m, "ObjectVector", pypv)
        .def(py::init<std::string, float, int>(), "name"_a, "mass"_a, "n_objects"_a=0);
    
    
    // Initial Conditions
    py::class_<InitialConditions> pyic(m, "InitialConditions");

    py::class_<UniformIC>(m, "UniformIC", pyic)
        .def(py::init<float>(), "density"_a);
        
    py::class_<RestartIC>(m, "RestartIC", pyic)
        .def(py::init<std::string>(),"path"_a = "restart/");
        
    py::class_<RigidIC>(m, "RigidIC", pyic)
        .def(py::init<std::string, std::string>(), "ic_filename"_a, "xyz_filename"_a);
        
    py::class_<MembraneIC>(m, "MembraneIC", pyic)
        .def(py::init<std::string, float>(), "ic_filename"_a, "global_scale"_a=1.0);
}
