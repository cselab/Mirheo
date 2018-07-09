#pragma once

#include "rigid_object_vector.h"

class RigidEllipsoidObjectVector : public RigidObjectVector
{
public:
    float3 axes;

    RigidEllipsoidObjectVector(std::string name, float mass,
                               const int objSize, float3 axes, const int nObjects = 0);

    virtual ~RigidEllipsoidObjectVector() {};
};


