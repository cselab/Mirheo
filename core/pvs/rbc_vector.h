#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include "object_vector.h"


class LocalRBCvector : public LocalObjectVector
{
public:
	LocalRBCvector(const int rbcSize, const int nRbcs = 0, cudaStream_t stream = 0) :
		LocalObjectVector(rbcSize, nRbcs)
	{
		// area and volumes need not to be communicated
		extraPerObject.createData<float2> ("area_volumes", nObjects);
	}

	virtual ~LocalRBCvector() = default;
};

class RBCvector : public ObjectVector
{
public:
	RBCvector(std::string name, float mass, const int objSize, Mesh mesh, const int nObjects = 0) :
		ObjectVector( name, mass, objSize,
					  new LocalRBCvector(objSize, nObjects),
					  new LocalRBCvector(objSize, 0) )
	{
		this->mesh = std::move(mesh);

		if (objSize != mesh.nvertices)
			die("RBC vector '%s': object size (%d) and number of vertices in mesh (%d) mismach",
					name.c_str(), objSize, mesh.nvertices);
	}

	LocalRBCvector* local() { return static_cast<LocalRBCvector*>(_local); }
	LocalRBCvector* halo()  { return static_cast<LocalRBCvector*>(_halo);  }

	virtual ~RBCvector() {};
};
