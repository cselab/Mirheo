#pragma once

#include "rigid_object_vector.h"

template <class Shape>
class RigidShapedObjectVector : public RigidObjectVector
{
public:
    RigidShapedObjectVector(const MirState *state, std::string name, real mass, int objSize,
                            Shape shape, int nObjects = 0);
    
    RigidShapedObjectVector(const MirState *state, std::string name, real mass, int objSize,
                            Shape shape, std::shared_ptr<Mesh> mesh, int nObjects = 0);
        
    virtual ~RigidShapedObjectVector();

    Shape shape;
};


