#pragma once

#include "rigid_object_vector.h"
#include <core/utils/pytypes.h>


class RigidEllipsoidObjectVector : public RigidObjectVector
{
public:
    float3 axes;

    RigidEllipsoidObjectVector(const YmrState *state, std::string name, float mass, const int objSize,
                               PyTypes::float3 axes, const int nObjects = 0);

    RigidEllipsoidObjectVector(const YmrState *state, std::string name, float mass, const int objSize,
                               PyTypes::float3 axes, std::shared_ptr<Mesh> mesh,
                               const int nObjects = 0);
        
    virtual ~RigidEllipsoidObjectVector();
};


