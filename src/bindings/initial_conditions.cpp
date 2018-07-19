#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/initial_conditions/interface.h>
#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/rigid_ic.h>
#include <core/initial_conditions/restart.h>
#include <core/initial_conditions/membrane_ic.h>

#include "nodelete.h"

namespace py = pybind11;
using namespace pybind11::literals;

void exportInitialConditions(py::module& m)
{
    // Initial Conditions
    py::nodelete_class<InitialConditions> pyic(m, "InitialConditions", R"(
            Base class for initial conditions
        )");

    py::nodelete_class<UniformIC>(m, "Uniform", pyic)
        .def(py::init<float>(), "density"_a, R"(
            The particles will be generated with the desired number density uniformly at random in all the domain.
            These IC may be used with any Particle Vector, but only make sense for regular PV.
            
            Args:
                density: target density
        )");
        
    py::nodelete_class<RestartIC>(m, "Restart", pyic)
        .def(py::init<std::string>(),"path"_a = "restart/", R"(
            Read the state (particle coordinates and velocities, other relevant data for objects **not implemented yet**)
        
            Args:
                path: folder where the restart files reside. The exact filename will be like this: <path>/<PV name>.chk
        )");
        
    py::nodelete_class<RigidIC>(m, "Rigid", pyic)
        .def(py::init<std::string, std::string>(), "ic_filename"_a, "xyz_filename"_a, R"(
            Can only be used with Rigid Object Vector or Rigid Ellipsoid, see :ref:`user-ic`. These IC will initialize the particles of each object
            according to the template .xyz file and then the objects will be translated/rotated according to the file initial conditions file.
            
            Args:
                ic_filename:
                    Text file describing location and rotation of the created objects.               
                    One line in the file corresponds to one object created.                          
                    Format of the line: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                xyz_filename:
                    Template that describes the positions of the body particles before translation or        
                    rotation is applied. Standard .xyz file format is used with first line being             
                    the number of particles, second comment, third and onwards - particle coordinates.       
                    The number of particles in the file should be the same as in number of particles per object
                    in the corresponding PV                                                        
        )");
        
    py::nodelete_class<MembraneIC>(m, "Membrane", pyic)
        .def(py::init<std::string, float>(), "ic_filename"_a, "global_scale"_a=1.0, R"(
            Can only be used with Membrane Object Vector, see :ref:`user-ic`. These IC will initialize the particles of each object
            according to the mesh associated with Membrane, and then the objects will be translated/rotated according to the file initial conditions file.
            
            Args:
                ic_filename:
                    Text file describing location and rotation of the created objects.               
                    One line in the file corresponds to one object created.                          
                    Format of the line: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized
                global_scale:
                    All the membranes will be scaled by that value. Useful to implement membranes growth so that they
                    can fill the space with high volume fraction                                        
        )");
}
