#include "rigid_ellipsoid_object_vector.h"
#include <core/utils/cuda_common.h>

RigidEllipsoidObjectVector::RigidEllipsoidObjectVector(
    std::string name, float mass,
    const int objSize, pyfloat3 axes, const int nObjects) :
        RigidObjectVector(
                name, mass,
                mass*objSize / 5.0f * make_float3(
                        sqr(std::get<1>(axes)) + sqr(std::get<2>(axes)),
                        sqr(std::get<2>(axes)) + sqr(std::get<0>(axes)),
                        sqr(std::get<0>(axes)) + sqr(std::get<1>(axes)) ),
                objSize,
                std::make_unique<Mesh>(), // TODO: need to generate ellipsoid mesh
                nObjects),
        axes(make_float3(axes))
{    }


