#pragma once

#include "rigid_object_vector.h"
#include <core/utils/pytypes.h>


class RigidEllipsoidObjectVector : public RigidObjectVector
{
public:
    float3 axes;

    RigidEllipsoidObjectVector(std::string name, float mass,
                               const int objSize, pyfloat3 axes, const int nObjects = 0);

    virtual ~RigidEllipsoidObjectVector() {};
};


