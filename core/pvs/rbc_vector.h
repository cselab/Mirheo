#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include "object_vector.h"

class RBCvector : public ObjectVector
{
public:
	RBCvector(std::string name, float mass, const int objSize, Mesh mesh, const int nObjects = 0) :
		ObjectVector( name, mass, objSize,
					  new LocalObjectVector(objSize, nObjects),
					  new LocalObjectVector(objSize, 0) )
	{
		this->mesh = std::move(mesh);

		if (objSize != mesh.nvertices)
			die("RBC vector '%s': object size (%d) and number of vertices in mesh (%d) mismach",
					name.c_str(), objSize, mesh.nvertices);

		requireDataPerObject<float2>("area_volumes", false);
	}

	virtual ~RBCvector() = default;
};
