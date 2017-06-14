#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/particle_vector.h>

class LocalObjectVector: public LocalParticleVector
{
protected:
	PinnedBuffer<int32_t*> _extraDataPtrs;
	PinnedBuffer<int>      _extraDataSizes;

public:

	struct __align__(16) COMandExtent
	{
		float3 com, low, high;
	};

	int nObjects = 0;
	int objSize  = 0;
	DeviceBuffer<int> particles2objIds;
	DeviceBuffer<COMandExtent> comAndExtents;

	LocalObjectVector(const int objSize, const int nObjects = 0) :
		LocalParticleVector(objSize*nObjects), objSize(objSize), nObjects(nObjects)
	{
		resize(nObjects*objSize, ResizeKind::resizeAnew);
		static_assert( sizeof(COMandExtent) % 4 == 0, "Extra data size in bytes should be divisible by 4" );

		_extraDataSizes.resize(1);
		_extraDataPtrs .resize(1);

		_extraDataSizes[0] = sizeof(COMandExtent) / 4;
		_extraDataPtrs [0] = (int32_t*)comAndExtents.devPtr();

		_extraDataSizes.uploadToDevice();
		_extraDataPtrs .uploadToDevice();
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

	// Provide some data per object for MPI exchange
	// ******************************************************************
	int extraDataNumPtrs()
	{
		return _extraDataSizes.size();
	}

	// Size in bytes
	// Device pointer!
	int* extraDataSizes()
	{
		return _extraDataSizes.devPtr();
	}

	// Device array of extraDataNumPtrs() device pointers
	int32_t** extraDataPtrs()
	{
		return _extraDataPtrs.devPtr();
	}
	// ******************************************************************

	void findExtentAndCOM(cudaStream_t stream);

	virtual ~LocalObjectVector() = default;
};


class ObjectVector : public ParticleVector
{
protected:
	ObjectVector(	std::string name, LocalObjectVector *local, LocalObjectVector *halo ) :
		ParticleVector(name, local, halo) {}

public:
	ObjectVector(std::string name, const int objSize, const int nObjects = 0) :
		ParticleVector( name,
						new LocalObjectVector(objSize, nObjects),
						new LocalObjectVector(objSize, nObjects) )
	{}

	LocalObjectVector* local() { return static_cast<LocalObjectVector*>(_local); }
	LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(_halo);  }

	virtual ~ObjectVector() {};
};
