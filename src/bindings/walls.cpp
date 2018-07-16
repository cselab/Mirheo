#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/walls/factory.h>

#include <core/utils/pytypes.h>

namespace py = pybind11;
using namespace pybind11::literals;


void exportWalls(py::module& m)
{
    // Initial Conditions
    py::class_<Wall> pywall(m, "Wall", R"(
        Base wall class.
        
        Walls are used to represent time-independent stationary boundary conditions for the flows. 
        Walls are represented in the form of a signed distance function (LINK), such that a zero-level isosurface defines the wall surface.
        No slip and no through boundary conditions are enforced on that surface by bouncing the particles off the wall surface.

        In order to prevent undesired density oscillations near the walls, so called frozen particles are used.
        These non-moving particles reside *inside* the walls and interact with the regular liquid particles.
        If the density and distribution of the frozen particles is the same as of the corresponding liquid particles,   
        the density oscillations in the liquid in proximity of the wall is minimal (LINK).
    
        In the beginning of the simulation all the particles define in the simulation (even not attached to the wall by class::`udevicex`) 
        will be checked against all the walls. Those inside the wall as well as objects partly inside the wall will be deleted.
        The only exception is the PVs that are named exactly as the wall, these PVs will be unaffected by their "parent" wall.
    )");

    py::class_< SimpleStationaryWall<StationaryWall_Box> >(m, "Box", pywall)
        .def(py::init(&WallFactory::createBoxWall),
            "name"_a, "low"_a, "high"_a, "inside"_a = false, R"(
            Rectangular cuboid wall with edges aligned with the coordinate axes.
            
            Args:
                name: name of the wall
                low: lower corner of the box
                high: higher corner of the box
                inside: whether the domain is inside the box or outside of it
        )");
        
    py::class_< SimpleStationaryWall<StationaryWall_Sphere> >(m, "Sphere", pywall)
        .def(py::init(&WallFactory::createSphereWall),
            "name"_a, "center"_a, "raduis"_a, "inside"_a = false, R"(
            Spherical wall.
            
            Args:
                name: name of the wall
                center: sphere center
                radius: sphere radius
                inside: whether the domain is inside the sphere or outside of it
        )");
        
    py::class_< SimpleStationaryWall<StationaryWall_Plane> >(m, "Plane", pywall)
        .def(py::init(&WallFactory::createPlaneWall),
            "name"_a, "normal"_a, "pointThrough"_a, R"(
            Planar infinitely stretching wall. Inside is determined by the normal direction .
            
            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
        )");
        
    py::class_< SimpleStationaryWall<StationaryWall_Cylinder> >(m, "Cylinder", pywall)
        .def(py::init(&WallFactory::createCylinderWall),
            "name"_a, "center"_a, "radius"_a, "axis"_a, "inside"_a = false, R"(
            Cylindrical infinitely stretching wall, the main axis is aligned along OX or OY or OZ
            
            Args:
                name: name of the wall
                center: point that belongs to the cylinder axis projected along that axis
                radius: cylinder radius
                axis: direction of cylinder axis, valid values are "x", "y" or "z"
                inside: whether the domain is inside the cylinder or outside of it
        )");
        
    py::class_< SimpleStationaryWall<StationaryWall_SDF> >(m, "SDF", pywall)
        .def(py::init(&WallFactory::createSDFWall),
            "name"_a, "sdfFilename"_a, "h"_a = pyfloat3{0.25, 0.25, 0.25}, R"(
            This wall is based on an arbitrary Signed Distance Function defined in the simulation domain on a regular cartesian grid.
            The wall reads the SDF data from a .sdf file, that has a special structure.
            
            First two lines define the header: three real number separated by spaces govern the size of the domain where the SDF is defined, 
            and next three integer numbers (:math:`Nx\,\,Ny\,\,Nz`) define the resolution.
            Next the :math:`Nx \times Ny \times Nz` single precision floating point values are written (in binary representation).
            
            Negative SDF values correspond to the domain, and positive -- to the inside of the wall.
            Threfore the boundary is defined by the zero-level isosurface.
            
            Args:
                name: name of the wall
                sdfFilename: lower corner of the box
                h: resolution of the resampled SDF. In order to have a more accurate SDF representation, the initial function is resampled on a finer grid. The lower this value is, the better the wall will be, however, the  more memory it will consume and the slower the execution will be
        )");
        
    py::class_< WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate> >(m, "RotatingCylinder", pywall)
        .def(py::init(&WallFactory::createMovingCylinderWall),
            "name"_a, "center"_a, "radius"_a, "axis"_a, "omega"_a, "inside"_a = false, R"(
            Cylindrical wall rotating with constant angular velocity along its axis.
            
            Args:
                name: name of the wall
                center: point that belongs to the cylinder axis projected along that axis
                radius: cylinder radius
                axis: direction of cylinder axis, valid values are "x", "y" or "z"
                omega: angular velocity of rotation along the cylinder axis
                inside: whether the domain is inside the cylinder or outside of it
        )");
        
    py::class_< WallWithVelocity<StationaryWall_Plane, VelocityField_Translate> >(m, "MovingPlane", pywall)
        .def(py::init(&WallFactory::createMovingPlaneWall),
            "name"_a, "normal"_a, "pointThrough"_a, "velocity"_a, R"(
            Planar wall that is moving along itself with constant velocity.
            Can be used to produce Couette velocity profile in combination with 
            The boundary conditions on such wall are no-through and constant velocity (specified).
            
            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
                velocity: wall velocity, should be orthogonal to the normal
        )");
        
    py::class_< WallWithVelocity<StationaryWall_Plane, VelocityField_Oscillate> >(m, "OscillatingPlane", pywall)
        .def(py::init(&WallFactory::createOscillatingPlaneWall),
            "name"_a, "normal"_a, "pointThrough"_a, "velocity"_a, "period"_a,  R"(
            Planar wall that is moving along itself with periodically changing velocity:
            
            .. math::
                \mathbf{u}(t) = cos(2*\pi * t / T);
            
            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
                velocity: velocity amplitude, should be orthogonal to the normal
                period: oscillation period in number of timesteps
        )");
}

