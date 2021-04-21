// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "rigid_ashape_object_vector.h"

#include <mirheo/core/analytical_shapes/api.h>

#include <memory>

namespace mirheo
{

namespace particle_vector_factory
{

inline std::shared_ptr<RigidShapedObjectVector<Capsule>>
createCapsuleROV(const MirState *state, const std::string& name, real mass, int objSize, real R, real L)
{
    Capsule cap(R, L);
    return std::make_shared<RigidShapedObjectVector<Capsule>>
        (state, name, mass, objSize, cap);
}

inline std::shared_ptr<RigidShapedObjectVector<Capsule>>
createCapsuleROVWithMesh(const MirState *state, const std::string& name, real mass, int objSize, real R, real L, std::shared_ptr<Mesh> mesh)
{
    Capsule cap(R, L);
    return std::make_shared<RigidShapedObjectVector<Capsule>>
        (state, name, mass, objSize, cap, std::move(mesh));
}



inline std::shared_ptr<RigidShapedObjectVector<Cylinder>>
createCylinderROV(const MirState *state, const std::string& name, real mass, int objSize, real R, real L)
{
    Cylinder cyl(R, L);
    return std::make_shared<RigidShapedObjectVector<Cylinder>>
        (state, name, mass, objSize, cyl);
}

inline std::shared_ptr<RigidShapedObjectVector<Cylinder>>
createCylinderROVWithMesh(const MirState *state, const std::string& name, real mass, int objSize, real R, real L, std::shared_ptr<Mesh> mesh)
{
    Cylinder cyl(R, L);
    return std::make_shared<RigidShapedObjectVector<Cylinder>>
        (state, name, mass, objSize, cyl, std::move(mesh));
}



inline std::shared_ptr<RigidShapedObjectVector<Ellipsoid>>
createEllipsoidROV(const MirState *state, const std::string& name, real mass, int objSize, real3 axes)
{
    Ellipsoid ell(axes);
    return std::make_shared<RigidShapedObjectVector<Ellipsoid>>
        (state, name, mass, objSize, ell);
}

inline std::shared_ptr<RigidShapedObjectVector<Ellipsoid>>
createEllipsoidROVWithMesh(const MirState *state, const std::string& name, real mass, int objSize, real3 axes, std::shared_ptr<Mesh> mesh)
{
    Ellipsoid ell(axes);
    return std::make_shared<RigidShapedObjectVector<Ellipsoid>>
        (state, name, mass, objSize, ell, std::move(mesh));
}

/** \brief Particle vector factory. Instantiate the correct vector type depending on the snapshot parameters.
    \param [in] state The global state of the system.
    \param [in] loader The \c Loader object. Provides load context and unserialization functions.
    \param [in] config The parameters of the particle vector.
 */
std::shared_ptr<ParticleVector> loadParticleVector(
        const MirState *state, Loader& loader, const ConfigObject& config);

} // namespace particle_vector_factory

} // namespace mirheo
