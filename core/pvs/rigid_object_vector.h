#pragma once

#include "object_vector.h"

#include <core/rigid_kernels/rigid_motion.h>

class RigidObjectVector : public ObjectVector
{
public:
	PinnedBuffer<float4> initialPositions;

	/// Diagonal of the inertia tensor in the principal axes
	/// The axes should be aligned with ox, oy, oz when q = {1 0 0 0}
	float3 J;

	RigidObjectVector(std::string name, float partMass, float3 J, const int objSize, const int nObjects = 0) :
		ObjectVector( name, partMass, objSize,
					  new LocalObjectVector(objSize, nObjects),
					  new LocalObjectVector(objSize, 0) ),
					  J(J)
	{
		// rigid motion must be exchanged and shifted
		requireDataPerObject<RigidMotion>("motions", true, sizeof(RigidReal));
	}

	virtual ~RigidObjectVector() = default;
};

#include "views/rov.h"



