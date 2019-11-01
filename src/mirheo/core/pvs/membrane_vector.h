#pragma once

#include "object_vector.h"

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/mesh/membrane.h>

class MembraneVector: public ObjectVector
{
public:
    MembraneVector(const MirState *state, std::string name, real mass, std::shared_ptr<MembraneMesh> mptr, int nObjects = 0);
    ~MembraneVector();
};
