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

	// Diagonal of the inertia tensor in the principal axes
	// The axes should be aligned with ox, oy, oz when q = {1 0 0 0}
	float3 J;

	RigidObjectVector(std::string name, float partMass, float3 J, const int objSize, const int nObjects = 0) :
		ObjectVector( name, partMass, objSize,
					  new LocalRigidObjectVector(objSize, nObjects),
					  new LocalRigidObjectVector(objSize, 0) ),
		J(J)
	{}

	LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(_local); }
	LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(_halo);  }

	virtual ~RigidObjectVector() {};
};

#include "views/rov.h"



