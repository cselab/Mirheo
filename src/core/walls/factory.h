#pragma once

#include "interface.h"

#include "simple_stationary_wall.h"
#include "wall_with_velocity.h"

#include "simple_stationary_wall.h"
#include "wall_with_velocity.h"

#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "stationary_walls/cylinder.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/box.h"

#include "velocity_field/rotate.h"
#include "velocity_field/translate.h"
#include "velocity_field/oscillate.h"

#include <core/utils/pytypes.h>
#include <core/utils/make_unique.h>

class ParticleVector;
class CellList;


class WallFactory
{
public:
    static SimpleStationaryWall<StationaryWall_Sphere>*
        createSphereWall(std::string name, pyfloat3 center, float radius, bool inside)
    {
        StationaryWall_Sphere sphere(make_float3(center), radius, inside);
        return new SimpleStationaryWall<StationaryWall_Sphere> (name, std::move(sphere));
    }

    static SimpleStationaryWall<StationaryWall_Box>*
        createBoxWall(std::string name, pyfloat3 low, pyfloat3 high, bool inside)
    {
        StationaryWall_Box box(make_float3(low), make_float3(high), inside);
        return new SimpleStationaryWall<StationaryWall_Box> (name, std::move(box));
    }

    static SimpleStationaryWall<StationaryWall_Cylinder>*
        createCylinderWall(std::string name, pyfloat2 center, float radius, std::string axis, bool inside)
    {
        StationaryWall_Cylinder::Direction dir;
        if (axis == "x") dir = StationaryWall_Cylinder::Direction::x;
        if (axis == "y") dir = StationaryWall_Cylinder::Direction::y;
        if (axis == "z") dir = StationaryWall_Cylinder::Direction::z;

        StationaryWall_Cylinder cylinder(make_float2(center), radius, dir, inside);
        return new SimpleStationaryWall<StationaryWall_Cylinder> (name, std::move(cylinder));
    }

    static SimpleStationaryWall<StationaryWall_Plane>* 
        createPlaneWall(std::string name, pyfloat3 normal, pyfloat3 pointThrough)
    {
        StationaryWall_Plane plane(normalize(make_float3(normal)), make_float3(pointThrough));
        return new SimpleStationaryWall<StationaryWall_Plane> (name, std::move(plane));
    }

    static SimpleStationaryWall<StationaryWall_SDF>*
        createSDFWall(std::string name, std::string sdfFilename, pyfloat3 h)
    {
        StationaryWall_SDF sdf(sdfFilename, make_float3(h));
        return new SimpleStationaryWall<StationaryWall_SDF> (name, std::move(sdf));
    }

    // Moving walls

    static WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate>*
        createMovingCylinderWall(std::string name, pyfloat2 _center, float radius, std::string axis, float omega, bool inside)
    {
        float2 center = make_float2(_center);
        
        StationaryWall_Cylinder::Direction dir;
        if (axis == "x") dir = StationaryWall_Cylinder::Direction::x;
        if (axis == "y") dir = StationaryWall_Cylinder::Direction::y;
        if (axis == "z") dir = StationaryWall_Cylinder::Direction::z;

        StationaryWall_Cylinder cylinder(center, radius, dir, inside);
        float3 omega3, center3;
		switch (dir)
		{
			case StationaryWall_Cylinder::Direction::x :
				center3 = {0.0f, center.x, center.y};
				omega3  = {omega,    0.0f,     0.0f};
				break;

			case StationaryWall_Cylinder::Direction::y :
				center3 = {center.x, 0.0f, center.y};
				omega3  = {0.0f,    omega,     0.0f};
				break;

			case StationaryWall_Cylinder::Direction::z :
				center3 = {center.x, center.y, 0.0f};
				omega3  = {0.0f,    0.0f,     omega};
				break;
		}
        VelocityField_Rotate rotate(omega3, center3);

        return new WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate> (name, std::move(cylinder), std::move(rotate));
    }

    static WallWithVelocity<StationaryWall_Plane, VelocityField_Translate>*
        createMovingPlaneWall(std::string name, pyfloat3 normal, pyfloat3 pointThrough, pyfloat3 velocity)
    {
        StationaryWall_Plane plane(normalize(make_float3(normal)), make_float3(pointThrough));
        VelocityField_Translate translate(make_float3(velocity));
        return new WallWithVelocity<StationaryWall_Plane, VelocityField_Translate> (name, std::move(plane), std::move(translate));
    }

    static WallWithVelocity<StationaryWall_Plane, VelocityField_Oscillate>*
        createOscillatingPlaneWall(std::string name, pyfloat3 normal, pyfloat3 pointThrough, pyfloat3 velocity, float period)
    {
        StationaryWall_Plane plane(normalize(make_float3(normal)), make_float3(pointThrough));
        VelocityField_Oscillate osc(make_float3(velocity), period);
        return new WallWithVelocity<StationaryWall_Plane, VelocityField_Oscillate> (name, std::move(plane), std::move(osc));
    }

};

