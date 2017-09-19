#pragma once

#include "object_vector.h"

class LocalRigidObjectVector: public LocalObjectVector
{
public:

	struct __align__(16) RigidMotion
	{
		float3 r; float4 q;
		float3 vel, omega;
		float3 force, torque;

		float4 prevQ;
	};

	PinnedBuffer<RigidMotion> motions;  // vector of com velocity, force and torque

	LocalRigidObjectVector(const int objSize, const int nObjects = 0, cudaStream_t stream = 0) :
		LocalObjectVector(objSize, nObjects)
	{
		resize(nObjects*objSize, stream, ResizeKind::resizeAnew);

		extraDataSizes.resize(2, stream);
		extraDataPtrs .resize(2, stream);

		extraDataSizes[0] = sizeof(COMandExtent);
		extraDataPtrs [0] = (char*)comAndExtents.devPtr();

		extraDataSizes[1] = sizeof(RigidMotion);
		extraDataPtrs [1] = (char*)motions.devPtr();

		extraDataSizes.uploadToDevice(stream);
		extraDataPtrs .uploadToDevice(stream);

		// Provide necessary alignment
		packedObjSize_bytes = ( (objSize*sizeof(Particle) + sizeof(COMandExtent) +sizeof(RigidMotion) + sizeof(float4)-1) / sizeof(float4) ) * sizeof(float4);
	}

	virtual void resize(const int np, cudaStream_t stream, ResizeKind kind = ResizeKind::resizePreserve)
	{
		LocalObjectVector::resize(np, stream, kind);
		motions.resize(nObjects, stream, kind);
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


