#pragma once

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

#include <memory>

namespace mirheo
{

class ParticleVector;
class CellList;


namespace WallFactory
{
inline std::shared_ptr<SimpleStationaryWall<StationaryWallSphere>>
createSphereWall(const MirState *state, const std::string& name, real3 center, real radius, bool inside)
{
    StationaryWallSphere sphere(center, radius, inside);
    return std::make_shared<SimpleStationaryWall<StationaryWallSphere>> (state, name, std::move(sphere));
}

inline std::shared_ptr<SimpleStationaryWall<StationaryWallBox>>
createBoxWall(const MirState *state, const std::string& name, real3 low, real3 high, bool inside)
{
    StationaryWallBox box(low, high, inside);
    return std::make_shared<SimpleStationaryWall<StationaryWallBox>> (state, name, std::move(box));
}

inline std::shared_ptr<SimpleStationaryWall<StationaryWallCylinder>>
createCylinderWall(const MirState *state, const std::string& name, real2 center, real radius, const std::string& axis, bool inside)
{
    StationaryWallCylinder::Direction dir;
    if (axis == "x") dir = StationaryWallCylinder::Direction::x;
    if (axis == "y") dir = StationaryWallCylinder::Direction::y;
    if (axis == "z") dir = StationaryWallCylinder::Direction::z;

    StationaryWallCylinder cylinder(center, radius, dir, inside);
    return std::make_shared<SimpleStationaryWall<StationaryWallCylinder>> (state, name, std::move(cylinder));
}

inline std::shared_ptr<SimpleStationaryWall<StationaryWallPlane>>
createPlaneWall(const MirState *state, const std::string& name, real3 normal, real3 pointThrough)
{
    StationaryWallPlane plane(normalize(normal), pointThrough);
    return std::make_shared<SimpleStationaryWall<StationaryWallPlane>> (state, name, std::move(plane));
}

inline std::shared_ptr<SimpleStationaryWall<StationaryWallSDF>>
createSDFWall(const MirState *state, const std::string& name, const std::string& sdfFilename, real3 h)
{
    StationaryWallSDF sdf(state, sdfFilename, h);
    return std::make_shared<SimpleStationaryWall<StationaryWallSDF>> (state, name, std::move(sdf));
}

// Moving walls

inline std::shared_ptr<WallWithVelocity<StationaryWallCylinder, VelocityFieldRotate>>
createMovingCylinderWall(const MirState *state, const std::string& name, real2 center, real radius, const std::string& axis, real omega, bool inside)
{
    StationaryWallCylinder::Direction dir;
    if (axis == "x") dir = StationaryWallCylinder::Direction::x;
    if (axis == "y") dir = StationaryWallCylinder::Direction::y;
    if (axis == "z") dir = StationaryWallCylinder::Direction::z;

    StationaryWallCylinder cylinder(center, radius, dir, inside);
    real3 omega3, center3;
    switch (dir)
    {
    case StationaryWallCylinder::Direction::x :
        center3 = {0.0_r, center.x, center.y};
        omega3  = {omega,    0.0_r,    0.0_r};
        break;

    case StationaryWallCylinder::Direction::y :
        center3 = {center.x, 0.0_r, center.y};
        omega3  = {0.0_r,    omega,    0.0_r};
        break;

    case StationaryWallCylinder::Direction::z :
        center3 = {center.x, center.y, 0.0_r};
        omega3  = {0.0_r,    0.0_r,    omega};
        break;
    }
    VelocityFieldRotate rotate(omega3, center3);

    return std::make_shared<WallWithVelocity<StationaryWallCylinder, VelocityFieldRotate>> (state, name, std::move(cylinder), std::move(rotate));
}

inline std::shared_ptr<WallWithVelocity<StationaryWallPlane, VelocityFieldTranslate>>
createMovingPlaneWall(const MirState *state, const std::string& name, real3 normal, real3 pointThrough, real3 velocity)
{
    StationaryWallPlane plane(normalize(normal), pointThrough);
    VelocityFieldTranslate translate(velocity);
    return std::make_shared<WallWithVelocity<StationaryWallPlane, VelocityFieldTranslate>> (state, name, std::move(plane), std::move(translate));
}

inline std::shared_ptr<WallWithVelocity<StationaryWallPlane, VelocityFieldOscillate>>
createOscillatingPlaneWall(const MirState *state, const std::string& name, real3 normal, real3 pointThrough, real3 velocity, real period)
{
    StationaryWallPlane plane(normalize(normal), pointThrough);
    VelocityFieldOscillate osc(velocity, period);
    return std::make_shared<WallWithVelocity<StationaryWallPlane, VelocityFieldOscillate>> (state, name, std::move(plane), std::move(osc));
}

/** \brief Wall factory. Instantiate the correct interaction object depending on the snapshot parameters.
    \param [in] state The global state of the system.
    \param [in] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] config The interaction parameters.
 */
std::shared_ptr<Wall>
loadWall(const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace WallFactory

} // namespace mirheo
