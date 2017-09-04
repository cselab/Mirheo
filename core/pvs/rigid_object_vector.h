#pragma once

#include <core/pvs/object_vector.h>

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
		static_assert( sizeof(RigidMotion)  % 4 == 0, "Extra data size in bytes should be divisible by 4" );


		extraDataSizes.resize(2, stream);
		extraDataPtrs .resize(2, stream);

		extraDataSizes[0] = sizeof(COMandExtent);
		extraDataPtrs [0] = (int32_t*)comAndExtents.devPtr();

		extraDataSizes[1] = sizeof(RigidMotion);
		extraDataPtrs [1] = (int32_t*)motions.devPtr();

		extraDataSizes.uploadToDevice(stream);
		extraDataPtrs .uploadToDevice(stream);

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
	float3 axes;
	PinnedBuffer<float4> initialPositions;

	RigidObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		ObjectVector( name, objSize,
					  new LocalRigidObjectVector(objSize, nObjects),
					  new LocalRigidObjectVector(objSize, 0) )
	{}

	LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(_local); }
	LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(_halo);  }

	virtual ~RigidObjectVector() {};
};


