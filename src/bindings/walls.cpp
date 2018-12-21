#include <core/walls/factory.h>

#include <core/utils/pytypes.h>

#include "bindings.h"
#include "class_wrapper.h"

using namespace pybind11::literals;

void exportWalls(py::module& m)
{
    py::handlers_class<Wall> pywall(m, "Wall", R"(
        Base wall class.
    )");
        
    pywall.def("attachFrozenParticles", &Wall::attachFrozen, R"(
        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    )");

    py::handlers_class< SimpleStationaryWall<StationaryWall_Box> >(m, "Box", pywall, R"(
        Rectangular cuboid wall with edges aligned with the coordinate axes.

    )")
        .def(py::init(&WallFactory::createBoxWall),
             "state"_a, "name"_a, "low"_a, "high"_a, "inside"_a = false, R"(
            Args:
                name: name of the wall
                low: lower corner of the box
                high: higher corner of the box
                inside: whether the domain is inside the box or outside of it
        )");
        
    py::handlers_class< SimpleStationaryWall<StationaryWall_Sphere> >(m, "Sphere", pywall, R"(
        Spherical wall.

    )")
        .def(py::init(&WallFactory::createSphereWall),
            "state"_a, "name"_a, "center"_a, "radius"_a, "inside"_a = false, R"(
            Args:
                name: name of the wall
                center: sphere center
                radius: sphere radius
                inside: whether the domain is inside the sphere or outside of it
        )");
        
    py::handlers_class< SimpleStationaryWall<StationaryWall_Plane> >(m, "Plane", pywall, R"(
        Planar infinitely stretching wall. Inside is determined by the normal direction .

    )")
        .def(py::init(&WallFactory::createPlaneWall),
            "state"_a, "name"_a, "normal"_a, "pointThrough"_a, R"(
            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
        )");
        
    py::handlers_class< SimpleStationaryWall<StationaryWall_Cylinder> >(m, "Cylinder", pywall, R"(
        Cylindrical infinitely stretching wall, the main axis is aligned along OX or OY or OZ
    )")
        .def(py::init(&WallFactory::createCylinderWall),
            "state"_a, "name"_a, "center"_a, "radius"_a, "axis"_a, "inside"_a = false, R"(
            Args:
                name: name of the wall
                center: point that belongs to the cylinder axis projected along that axis
                radius: cylinder radius
                axis: direction of cylinder axis, valid values are "x", "y" or "z"
                inside: whether the domain is inside the cylinder or outside of it
        )");
        
    py::handlers_class< SimpleStationaryWall<StationaryWall_SDF> >(m, "SDF", pywall, R"(
        This wall is based on an arbitrary Signed Distance Function defined in the simulation domain on a regular cartesian grid.
        The wall reads the SDF data from a .sdf file, that has a special structure.
        
        First two lines define the header: three real number separated by spaces govern the size of the domain where the SDF is defined, 
        and next three integer numbers (:math:`Nx\,\,Ny\,\,Nz`) define the resolution.
        Next the :math:`Nx \times Ny \times Nz` single precision floating point values are written (in binary representation).
        
        Negative SDF values correspond to the domain, and positive -- to the inside of the wall.
        Therefore the boundary is defined by the zero-level isosurface.
    )")
        .def(py::init(&WallFactory::createSDFWall),
            "state"_a, "name"_a, "sdfFilename"_a, "h"_a = PyTypes::float3{0.25, 0.25, 0.25}, R"(
            Args:
                name: name of the wall
                sdfFilename: lower corner of the box
                h: resolution of the resampled SDF. In order to have a more accurate SDF representation, the initial function is resampled on a finer grid. The lower this value is, the better the wall will be, however, the  more memory it will consume and the slower the execution will be
        )");
        
    py::handlers_class< WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate> >(m, "RotatingCylinder", pywall, R"(
        Cylindrical wall rotating with constant angular velocity along its axis.
    )")
        .def(py::init(&WallFactory::createMovingCylinderWall),
            "state"_a, "name"_a, "center"_a, "radius"_a, "axis"_a, "omega"_a, "inside"_a = false, R"(
            Args:
                name: name of the wall
                center: point that belongs to the cylinder axis projected along that axis
                radius: cylinder radius
                axis: direction of cylinder axis, valid values are "x", "y" or "z"
                omega: angular velocity of rotation along the cylinder axis
                inside: whether the domain is inside the cylinder or outside of it
        )");
        
    py::handlers_class< WallWithVelocity<StationaryWall_Plane, VelocityField_Translate> >(m, "MovingPlane", pywall, R"(
        Planar wall that is moving along itself with constant velocity.
        Can be used to produce Couette velocity profile in combination with 
        The boundary conditions on such wall are no-through and constant velocity (specified).
    )")
        .def(py::init(&WallFactory::createMovingPlaneWall),
            "state"_a, "name"_a, "normal"_a, "pointThrough"_a, "velocity"_a, R"(
            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
                velocity: wall velocity, should be orthogonal to the normal
        )");
        
    py::handlers_class< WallWithVelocity<StationaryWall_Plane, VelocityField_Oscillate> >(m, "OscillatingPlane", pywall, R"(
        Planar wall that is moving along itself with periodically changing velocity:
        
        .. math::
            \mathbf{u}(t) = cos(2*\pi * t / T); 
    )")
        .def(py::init(&WallFactory::createOscillatingPlaneWall),
            "state"_a, "name"_a, "normal"_a, "pointThrough"_a, "velocity"_a, "period"_a,  R"(
            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
                velocity: velocity amplitude, should be orthogonal to the normal
                period: oscillation period dpd time units
        )");
}

