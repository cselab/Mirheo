#pragma once

#include "rigid_object_vector.h"

class RigidEllipsoidObjectVector : public RigidObjectVector
{
public:
	float3 axes;

	RigidEllipsoidObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		RigidObjectVector(name, objSize, nObjects)
	{}

	LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(_local); }
	LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(_halo);  }

	virtual ~RigidEllipsoidObjectVector() {};
};

