#pragma once

#include "particle_vector.h"

#include <core/containers.h>
#include <core/logger.h>
#include <core/mesh.h>

class LocalObjectVector: public LocalParticleVector
{
protected:
	int objSize  = 0;

public:
	int nObjects = 0;

	bool comExtentValid = false;

	ExtraDataManager extraPerObject;

	struct __align__(16) COMandExtent
	{
		float3 com, low, high;
	};


	LocalObjectVector(const int objSize, const int nObjects = 0, cudaStream_t stream = 0) :
		LocalParticleVector(objSize*nObjects), objSize(objSize), nObjects(nObjects)
	{
		resize(nObjects*objSize, stream);

		// center of mass and extents are not to be sent around
		// it's cheaper to compute them on site
		extraPerObject.createData<COMandExtent> ("com_extents", nObjects);

		// object ids must always follow objects
		extraPerObject.createData<int> ("ids", nObjects);
		extraPerObject.needExchange("ids");
	};


	void resize(const int np, cudaStream_t stream) override
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;
		LocalParticleVector::resize(np, stream);

		extraPerObject.resize(np, stream);
	}

	void resize_anew(const int np) override
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;
		LocalParticleVector::resize_anew(np);

		extraPerObject.resize_anew(np);
	}

	virtual PinnedBuffer<Particle>* getMeshVertices(cudaStream_t stream)
	{
		return &coosvels;
	}


	virtual ~LocalObjectVector() = default;
};


class ObjectVector : public ParticleVector
{
protected:
	ObjectVector( std::string name, float mass, int objSize, LocalObjectVector *local, LocalObjectVector *halo ) :
		ParticleVector(name, mass, local, halo), objSize(objSize) {}

public:
	int objSize;
	Mesh mesh;

	ObjectVector(std::string name, float mass, const int objSize, const int nObjects = 0) :
		ObjectVector( name, mass, objSize,
					  new LocalObjectVector(objSize, nObjects),
					  new LocalObjectVector(objSize, 0) )
	{}

	void findExtentAndCOM(cudaStream_t stream, bool isLocal);

	LocalObjectVector* local() { return static_cast<LocalObjectVector*>(_local); }
	LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(_halo);  }

	virtual ~ObjectVector() = default;
};

#include "views/ov.h"




