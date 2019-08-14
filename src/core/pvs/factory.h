#pragma once

#include "rigid_ashape_object_vector.h"

#include <core/analytical_shapes/api.h>
#include <core/utils/pytypes.h>

#include <memory>

namespace ParticleVectorFactory
{

inline std::shared_ptr<RigidShapedObjectVector<Capsule>>
createCapsuleROV(const MirState *state, std::string name, float mass, int objSize, float R, float L)
{
    Capsule cap(R, L);
    return std::make_shared<RigidShapedObjectVector<Capsule>>
        (state, name, mass, objSize, cap);
}

inline std::shared_ptr<RigidShapedObjectVector<Capsule>>
createCapsuleROVWithMesh(const MirState *state, std::string name, float mass, int objSize, float R, float L, std::shared_ptr<Mesh> mesh)
{
    Capsule cap(R, L);
    return std::make_shared<RigidShapedObjectVector<Capsule>>
        (state, name, mass, objSize, cap, std::move(mesh));
}



inline std::shared_ptr<RigidShapedObjectVector<Cylinder>>
createCylinderROV(const MirState *state, std::string name, float mass, int objSize, float R, float L)
{
    Cylinder cyl(R, L);
    return std::make_shared<RigidShapedObjectVector<Cylinder>>
        (state, name, mass, objSize, cyl);
}

inline std::shared_ptr<RigidShapedObjectVector<Cylinder>>
createCylinderROVWithMesh(const MirState *state, std::string name, float mass, int objSize, float R, float L, std::shared_ptr<Mesh> mesh)
{
    Cylinder cyl(R, L);
    return std::make_shared<RigidShapedObjectVector<Cylinder>>
        (state, name, mass, objSize, cyl, std::move(mesh));
}



inline std::shared_ptr<RigidShapedObjectVector<Ellipsoid>>
createEllipsoidROV(const MirState *state, std::string name, float mass, int objSize, PyTypes::float3 axes)
{
    Ellipsoid ell(make_float3(axes));
    return std::make_shared<RigidShapedObjectVector<Ellipsoid>>
        (state, name, mass, objSize, ell);
}

inline std::shared_ptr<RigidShapedObjectVector<Ellipsoid>>
createEllipsoidROVWithMesh(const MirState *state, std::string name, float mass, int objSize, PyTypes::float3 axes, std::shared_ptr<Mesh> mesh)
{
    Ellipsoid ell(make_float3(axes));
    return std::make_shared<RigidShapedObjectVector<Ellipsoid>>
        (state, name, mass, objSize, ell, std::move(mesh));
}

} // namespace ParticleVectorFactory
