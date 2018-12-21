#include "rigid_ellipsoid_object_vector.h"
#include <core/utils/cuda_common.h>

static float3 inertia_tensor(float mass, int objsize, float3 axes)
{
    return mass * objsize / 5.0f * make_float3
        (sqr(axes.y) + sqr(axes.z),
         sqr(axes.x) + sqr(axes.z),
         sqr(axes.x) + sqr(axes.y));
}

RigidEllipsoidObjectVector::RigidEllipsoidObjectVector(
        const YmrState *state, std::string name, float mass, const int objSize,
        PyTypes::float3 axes, const int nObjects) :
    RigidObjectVector(state, name, mass,
                      inertia_tensor(mass, objSize, make_float3(axes)),
                      objSize,
                      std::make_shared<Mesh>(),
                      nObjects),
    axes(make_float3(axes))
{}


RigidEllipsoidObjectVector::RigidEllipsoidObjectVector(
        const YmrState *state, std::string name, float mass, const int objSize,
        PyTypes::float3 axes, std::shared_ptr<Mesh> mesh,
        const int nObjects) :
    RigidObjectVector(state, name, mass,
                      inertia_tensor(mass, objSize, make_float3(axes)),
                      objSize, mesh, nObjects),
    axes(make_float3(axes))
{}

RigidEllipsoidObjectVector::~RigidEllipsoidObjectVector() = default;
