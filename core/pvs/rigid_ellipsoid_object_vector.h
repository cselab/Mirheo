#pragma once

#include "rigid_object_vector.h"

class LocalRigidEllipsoidObjectVector: public LocalRigidObjectVector
{
public:

	LocalRigidEllipsoidObjectVector(const int objSize, const int nObjects = 0, cudaStream_t stream = 0) :
		LocalRigidObjectVector(objSize, nObjects)
	{ }

	virtual ~LocalRigidEllipsoidObjectVector() = default;
};

class RigidEllipsoidObjectVector : public RigidObjectVector
{
public:
	float3 axes;

	RigidEllipsoidObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		RigidObjectVector(name, objSize, nObjects)
	{}


	LocalRigidEllipsoidObjectVector* local() { return static_cast<LocalRigidEllipsoidObjectVector*>(_local); }
	LocalRigidEllipsoidObjectVector* halo()  { return static_cast<LocalRigidEllipsoidObjectVector*>(_halo);  }

	virtual ~RigidEllipsoidObjectVector() {};
};

