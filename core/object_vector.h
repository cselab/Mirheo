#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/particle_vector.h>

class CellList;

struct ObjectVector: public ParticleVector
{
	struct __align__(16) Properties
	{
		float3 com, high, low;
	};


	int nObjects = 0;
	int objSize  = 0;
	PinnedBuffer<int> objStarts, objSizes;
	DeviceBuffer<int> particles2objIds;
	DeviceBuffer<Properties> properties;

	PinnedBuffer<Force> haloForces;
	DeviceBuffer<int>   haloIds;


	virtual void pushStreamWOhalo(cudaStream_t stream)
	{
		ParticleVector::pushStreamWOhalo(stream);
		objStarts.		 pushStream(stream);
		objSizes.		 pushStream(stream);
		particles2objIds.pushStream(stream);
		properties.		 pushStream(stream);
	}

	virtual void popStreamWOhalo()
	{
		ParticleVector::popStreamWOhalo();
		objStarts.		 popStream();
		objSizes.		 popStream();
		particles2objIds.popStream();
		properties.		 popStream();
	}

	virtual void resize(const int nObj, ResizeKind kind = ResizeKind::resizePreserve)
	{
		ParticleVector::resize(nObj * objSize, kind);
		particles2objIds.resize(np, kind);
		objStarts. resize(nObj+1);
		objSizes.  resize(nObj);
		properties.resize(nObj);
	}

	virtual ~ObjectVector() = default;

	void findExtentAndCOM(cudaStream_t stream);
};
