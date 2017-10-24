#pragma once

#include "object_vector.h"

#include <core/rigid_kernels/rigid_motion.h>

class LocalRigidObjectVector: public LocalObjectVector
{
public:

	LocalRigidObjectVector(const int objSize, const int nObjects = 0, cudaStream_t stream = 0) :
		LocalObjectVector(objSize, nObjects)
	{
		dataPerObject["motions"] = std::make_unique< PinnedBuffer<RigidMotion> >(nObjects);
	}

	virtual ~LocalRigidObjectVector() = default;
};

class RigidObjectVector : public ObjectVector
{
public:
	PinnedBuffer<float4> initialPositions;

	RigidObjectVector(std::string name, float mass, const int objSize, const int nObjects = 0) :
		ObjectVector( name, mass, objSize,
					  new LocalRigidObjectVector(objSize, nObjects),
					  new LocalRigidObjectVector(objSize, 0) )
	{}

	virtual float3 getInertiaTensor() { return make_float3(1); }

	LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(_local); }
	LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(_halo);  }

	virtual ~RigidObjectVector() {};
};

#include "views/rov.h"



