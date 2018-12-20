#pragma once

#include <memory>

#include <core/utils/make_unique.h>
#include <core/utils/pytypes.h>

#include "interface.h"

#include "simple_stationary_wall.h"
#include "stationary_walls/box.h"
#include "stationary_walls/cylinder.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "velocity_field/oscillate.h"
#include "velocity_field/rotate.h"
#include "velocity_field/translate.h"
#include "wall_with_velocity.h"

class ParticleVector;
class CellList;


namespace WallFactory
{
    static std::shared_ptr<SimpleStationaryWall<StationaryWall_Sphere>>
    createSphereWall(std::string name, PyTypes::float3 center, float radius, bool inside)
    {
        StationaryWall_Sphere sphere(make_float3(center), radius, inside);
        return std::make_shared<SimpleStationaryWall<StationaryWall_Sphere>> (name, std::move(sphere));
    }

    static std::shared_ptr<SimpleStationaryWall<StationaryWall_Box>>
    createBoxWall(std::string name, PyTypes::float3 low, PyTypes::float3 high, bool inside)
    {
        StationaryWall_Box box(make_float3(low), make_float3(high), inside);
        return std::make_shared<SimpleStationaryWall<StationaryWall_Box>> (name, std::move(box));
    }

    static std::shared_ptr<SimpleStationaryWall<StationaryWall_Cylinder>>
    createCylinderWall(std::string name, PyTypes::float2 center, float radius, std::string axis, bool inside)
    {
        StationaryWall_Cylinder::Direction dir;
        if (axis == "x") dir = StationaryWall_Cylinder::Direction::x;
        if (axis == "y") dir = StationaryWall_Cylinder::Direction::y;
        if (axis == "z") dir = StationaryWall_Cylinder::Direction::z;

        StationaryWall_Cylinder cylinder(make_float2(center), radius, dir, inside);
        return std::make_shared<SimpleStationaryWall<StationaryWall_Cylinder>> (name, std::move(cylinder));
    }

    static std::shared_ptr<SimpleStationaryWall<StationaryWall_Plane>>
    createPlaneWall(std::string name, PyTypes::float3 normal, PyTypes::float3 pointThrough)
    {
        StationaryWall_Plane plane(normalize(make_float3(normal)), make_float3(pointThrough));
        return std::make_shared<SimpleStationaryWall<StationaryWall_Plane>> (name, std::move(plane));
    }

    static std::shared_ptr<SimpleStationaryWall<StationaryWall_SDF>>
    createSDFWall(std::string name, std::string sdfFilename, PyTypes::float3 h)
    {
        StationaryWall_SDF sdf(sdfFilename, make_float3(h));
        return std::make_shared<SimpleStationaryWall<StationaryWall_SDF>> (name, std::move(sdf));
    }

    // Moving walls

    static std::shared_ptr<WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate>>
    createMovingCylinderWall(std::string name, PyTypes::float2 _center, float radius, std::string axis, float omega, bool inside)
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

        return std::make_shared<WallWithVelocity<StationaryWall_Cylinder, VelocityField_Rotate>> (name, std::move(cylinder), std::move(rotate));
    }

    static std::shared_ptr<WallWithVelocity<StationaryWall_Plane, VelocityField_Translate>>
    createMovingPlaneWall(std::string name, PyTypes::float3 normal, PyTypes::float3 pointThrough, PyTypes::float3 velocity)
    {
        StationaryWall_Plane plane(normalize(make_float3(normal)), make_float3(pointThrough));
        VelocityField_Translate translate(make_float3(velocity));
        return std::make_shared<WallWithVelocity<StationaryWall_Plane, VelocityField_Translate>> (name, std::move(plane), std::move(translate));
    }

    static std::shared_ptr<WallWithVelocity<StationaryWall_Plane, VelocityField_Oscillate>>
    createOscillatingPlaneWall(std::string name, PyTypes::float3 normal, PyTypes::float3 pointThrough, PyTypes::float3 velocity, float period)
    {
        StationaryWall_Plane plane(normalize(make_float3(normal)), make_float3(pointThrough));
        VelocityField_Oscillate osc(make_float3(velocity), period);
        return std::make_shared<WallWithVelocity<StationaryWall_Plane, VelocityField_Oscillate>> (name, std::move(plane), std::move(osc));
    }
};

