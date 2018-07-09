#include "rigid_ellipsoid_object_vector.h"

RigidEllipsoidObjectVector::RigidEllipsoidObjectVector(
    std::string name, float mass,
    const int objSize, float3 axes, const int nObjects) :
        RigidObjectVector(
                name, mass,
                mass*objSize / 5.0f * make_float3(
                        axes.y*axes.y + axes.z*axes.z,
                        axes.z*axes.z + axes.x*axes.x,
                        axes.x*axes.x + axes.y*axes.y ),
                objSize,
                std::make_unique<Mesh>(), // TODO: need to generate ellipsoid mesh
                nObjects),
        axes(axes)
{    }


