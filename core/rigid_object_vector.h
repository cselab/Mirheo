#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/object_vector.h>

class LocalRigidObjectVector: public LocalObjectVector
{
public:

	struct __align__(16) RigidMotion
	{
		float3 r; float4 q;
		float3 vel, omega;
		float3 force, torque;

		// Track the changes in the object position and rotation
		// Used in updating the properties of the constituting particles
		float3 deltaR;
		float4 deltaQ;
		float3 deltaV;
		float3 deltaW;
	};

	PinnedBuffer<RigidMotion> motions;  // vector of com velocity, force and torque

	LocalRigidObjectVector(const int objSize, const int nObjects = 0) :
		LocalObjectVector(objSize, nObjects)
	{
		resize(nObjects*objSize, ResizeKind::resizeAnew);
		static_assert( sizeof(COMandExtent) % 4 == 0, "Extra data size in bytes should be divisible by 4" );
		static_assert( sizeof(RigidMotion)  % 4 == 0, "Extra data size in bytes should be divisible by 4" );


		extraDataSizes.resize(2);
		extraDataPtrs .resize(2);

		extraDataSizes[0] = sizeof(COMandExtent);
		extraDataPtrs [0] = (int32_t*)comAndExtents.devPtr();

		extraDataSizes[1] = sizeof(RigidMotion);
		extraDataPtrs [1] = (int32_t*)motions.devPtr();

		extraDataSizes.uploadToDevice();
		extraDataPtrs .uploadToDevice();

		packedObjSize_bytes = ( (objSize*sizeof(Particle) + sizeof(COMandExtent) +sizeof(RigidMotion) + sizeof(float4)-1) / sizeof(float4) ) * sizeof(float4);
	}


	virtual void pushStream(cudaStream_t stream)
	{
		LocalObjectVector::pushStream(stream);
		motions.pushStream(stream);
	}

	virtual void popStream()
	{
		LocalObjectVector::popStream();
		motions.popStream();
	}

	virtual void resize(const int np, ResizeKind kind = ResizeKind::resizePreserve)
	{
		LocalObjectVector::resize(np, kind);
		motions.resize(nObjects, kind);
	}

	virtual ~LocalRigidObjectVector() = default;
};

class RigidObjectVector : public ObjectVector
{
public:
	float3 axes;

	RigidObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		ObjectVector( name, objSize,
					  new LocalRigidObjectVector(objSize, nObjects),
					  new LocalRigidObjectVector(objSize, 0) )
	{}

	LocalRigidObjectVector* local() { return static_cast<LocalRigidObjectVector*>(_local); }
	LocalRigidObjectVector* halo()  { return static_cast<LocalRigidObjectVector*>(_halo);  }

	virtual ~RigidObjectVector() {};
};


