#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/object_vector.h>

class CellList;

struct RigidObjectVector: public ObjectVector
{
	struct __align__(16) RigidMovement
	{
		float3 vel, omega, force, torque;
	};

	DeviceBuffer<RigidMovement> vs_fs_ts;  // vector of com velocity, force and torque


	RigidObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		ObjectVector(name, objSize, nObjects) { }


	virtual void pushStreamWOhalo(cudaStream_t stream)
	{
		ObjectVector::pushStreamWOhalo(stream);

		vs_fs_ts.pushStream(stream);
	}

	virtual void popStreamWOhalo()
	{
		ObjectVector::popStreamWOhalo();

		vs_fs_ts.popStream();
	}

	virtual void resize(const int np, ResizeKind kind = ResizeKind::resizePreserve)
	{
		ObjectVector::resize(np, kind);
		vs_fs_ts.resize(nObjects, kind);
	}

	virtual ~RigidObjectVector() = default;
};
