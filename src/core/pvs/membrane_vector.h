#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include "object_vector.h"

class MembraneVector: public ObjectVector
{
public:
    MembraneVector(std::string name, float mass, const int objSize, std::shared_ptr<MembraneMesh> mptr, const int nObjects = 0) :
        ObjectVector( name, mass, objSize,
                      new LocalObjectVector(this, objSize, nObjects),
                      new LocalObjectVector(this, objSize, 0) )
    {
        mesh = std::move(mptr);

        if (objSize != mesh->nvertices)
            die("RBC vector '%s': object size (%d) and number of vertices in mesh (%d) mismatch",
                    name.c_str(), objSize, mesh->nvertices);

        requireDataPerObject<float2>("area_volumes", false);
    }

    virtual ~MembraneVector() = default;
};
