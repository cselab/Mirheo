#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <core/initial_conditions/interface.h>
#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/rigid_ic.h>
#include <core/initial_conditions/restart.h>
#include <core/initial_conditions/membrane_ic.h>
#include <core/initial_conditions/from_array.h>

#include <core/utils/pytypes.h>

#include "nodelete.h"

namespace py = pybind11;
using namespace pybind11::literals;

void exportInitialConditions(py::module& m)
{
    // Initial Conditions
    py::handlers_class<InitialConditions> pyic(m, "InitialConditions", R"(
            Base class for initial conditions
        )");

    py::handlers_class<UniformIC>(m, "Uniform", pyic, R"(
        The particles will be generated with the desired number density uniformly at random in all the domain.
        These IC may be used with any Particle Vector, but only make sense for regular PV.
            
    )")
        .def(py::init<float>(), "density"_a, R"(
            Args:
                density: target density
        )");
        
    py::handlers_class<RestartIC>(m, "Restart", pyic, R"(
        Read the state (particle coordinates and velocities, other relevant data for objects is **not implemented yet**)
    )")
        .def(py::init<std::string>(),"path"_a = "restart/", R"(

            Args:
                path: folder where the restart files reside. The exact filename will be like this: <path>/<PV name>.chk
        )");
        
    py::handlers_class<RigidIC>(m, "Rigid", pyic, R"(
        Can only be used with Rigid Object Vector or Rigid Ellipsoid, see :ref:`user-ic`. These IC will initialize the particles of each object
        according to the template .xyz file and then the objects will be translated/rotated according to the provided initial conditions.
            
    )")
        .def(py::init<ICvector, std::string>(), "com_q"_a, "xyz_filename"_a, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                xyz_filename:
                    Template that describes the positions of the body particles before translation or        
                    rotation is applied. Standard .xyz file format is used with first line being             
                    the number of particles, second comment, third and onwards - particle coordinates.       
                    The number of particles in the file must be the same as in number of particles per object
                    in the corresponding PV
        )")
        .def(py::init<ICvector, const PyContainer&>(), "com_q"_a, "coords"_a, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                coords:
                    Template that describes the positions of the body particles before translation or        
                    rotation is applied.       
                    The number of coordinates must be the same as in number of particles per object
                    in the corresponding PV
        )");
    
        
    py::handlers_class<MembraneIC>(m, "Membrane", pyic, R"(
        Can only be used with Membrane Object Vector, see :ref:`user-ic`. These IC will initialize the particles of each object
        according to the mesh associated with Membrane, and then the objects will be translated/rotated according to the provided initial conditions.
    )")
        .def(py::init<ICvector, float>(), "com_q"_a, "global_scale"_a=1.0, R"(
            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                global_scale:
                    All the membranes will be scaled by that value. Useful to implement membranes growth so that they
                    can fill the space with high volume fraction                                        
        )");

    py::handlers_class<FromArrayIC>(m, "FromArray", pyic, R"(
        Set particles according to given position and velocity arrays.            
    )")
        .def(py::init<const PyContainer&, const PyContainer&>(), "pos"_a, "vel"_a, R"(
            Args:
                pos: array of positions
                vel: array of velocities
        )");
        

}
