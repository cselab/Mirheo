#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/object_vector.h>

class CellList;

struct RigidObjectVector: public ObjectVector
{
	struct __align__(16) RigidMotion
	{
		float4 q;
		float3 vel, omega;
		float3 force, torque;
	};

	DeviceBuffer<RigidMotion> motion;  // vector of com velocity, force and torque


	RigidObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		ObjectVector(name, objSize, nObjects) { }


	virtual void pushStreamWOhalo(cudaStream_t stream)
	{
		ObjectVector::pushStreamWOhalo(stream);

		motion.pushStream(stream);
	}

	virtual void popStreamWOhalo()
	{
		ObjectVector::popStreamWOhalo();

		motion.popStream();
	}

	virtual void resize(const int np, ResizeKind kind = ResizeKind::resizePreserve)
	{
		ObjectVector::resize(np, kind);
		motion.resize(nObjects, kind);
	}

	virtual ~RigidObjectVector() = default;
};
