#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/particle_vector.h>

class LocalObjectVector: public LocalParticleVector
{
public:

	PinnedBuffer<int32_t*> extraDataPtrs;
	PinnedBuffer<int>      extraDataSizes;

	struct __align__(16) COMandExtent
	{
		float3 com, low, high;
	};

	int nObjects = 0;
	int objSize  = 0;
	int packedObjSize_bytes = 0;
	DeviceBuffer<int> particles2objIds;
	DeviceBuffer<COMandExtent> comAndExtents;

	LocalObjectVector(const int objSize, const int nObjects = 0) :
		LocalParticleVector(objSize*nObjects), objSize(objSize), nObjects(nObjects)
	{
		resize(nObjects*objSize, ResizeKind::resizeAnew);
		static_assert( sizeof(COMandExtent) % 4 == 0, "Extra data size in bytes should be divisible by 4" );

		extraDataSizes.resize(1);
		extraDataPtrs .resize(1);

		extraDataSizes[0] = sizeof(COMandExtent);
		extraDataPtrs [0] = (int32_t*)comAndExtents.devPtr();

		extraDataSizes.uploadToDevice();
		extraDataPtrs .uploadToDevice();

		packedObjSize_bytes = ( (objSize*sizeof(Particle) + sizeof(COMandExtent) + sizeof(float4)-1) / sizeof(float4) ) * sizeof(float4);
	};

	virtual void pushStream(cudaStream_t stream)
	{
		LocalParticleVector::pushStream(stream);

		particles2objIds.pushStream(stream);
		comAndExtents   .pushStream(stream);
	}

	virtual void popStream()
	{
		LocalParticleVector::popStream();

		particles2objIds.popStream();
		comAndExtents   .popStream();
	}

	virtual void resize(const int np, ResizeKind kind = ResizeKind::resizePreserve)
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;

		LocalParticleVector::resize(nObjects * objSize, kind);
		particles2objIds.resize(np,       kind);
		comAndExtents   .resize(nObjects, kind);
	}

	void findExtentAndCOM(cudaStream_t stream);

	virtual ~LocalObjectVector() = default;
};


class ObjectVector : public ParticleVector
{
protected:
	ObjectVector( std::string name, int objSize, LocalObjectVector *local, LocalObjectVector *halo ) :
		ParticleVector(name, local, halo), objSize(objSize) {}

public:
	int objSize;
	float objMass;

	ObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		ObjectVector( name, objSize,
					  new LocalObjectVector(objSize, nObjects),
					  new LocalObjectVector(objSize, 0) )
	{}

	LocalObjectVector* local() { return static_cast<LocalObjectVector*>(_local); }
	LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(_halo);  }

	virtual ~ObjectVector() {};
};
