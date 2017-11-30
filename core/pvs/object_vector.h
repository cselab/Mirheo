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
	};

	void resize(const int np, cudaStream_t stream) override
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;
		LocalParticleVector::resize(np, stream);

		extraPerObject.resize(nObjects, stream);
	}

	void resize_anew(const int np) override
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;
		LocalParticleVector::resize_anew(np);

		extraPerObject.resize_anew(nObjects);
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
		ParticleVector(name, mass, local, halo), objSize(objSize)
	{
		// center of mass and extents are not to be sent around
		// it's cheaper to compute them on site
		requireDataPerObject<LocalObjectVector::COMandExtent>("com_extents", false);

		// object ids must always follow objects
		requireDataPerObject<int>("ids", true);
	}

public:
	int objSize;
	Mesh mesh;

	ObjectVector(std::string name, float mass, const int objSize, const int nObjects = 0) :
		ObjectVector( name, mass, objSize,
					  new LocalObjectVector(objSize, nObjects),
					  new LocalObjectVector(objSize, 0) )
	{	}

	void findExtentAndCOM(cudaStream_t stream, bool isLocal);

	LocalObjectVector* local() { return static_cast<LocalObjectVector*>(_local); }
	LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(_halo);  }

	template<typename T>
	void requireDataPerObject(std::string name, bool needExchange)
	{
		requireDataPerObject<T>(name, needExchange, 0);
	}

	template<typename T>
	void requireDataPerObject(std::string name, bool needExchange, int shiftDataType)
	{
		requireDataPerObject<T>(local(), name, needExchange, shiftDataType);
		requireDataPerObject<T>(halo(),  name, needExchange, shiftDataType);
	}

	virtual ~ObjectVector() = default;

private:
	template<typename T>
	void requireDataPerObject(LocalObjectVector* lov, std::string name, bool needExchange, int shiftDataType)
	{
		lov->extraPerObject.createData<T> (name, lov->nObjects);
		if (needExchange) lov->extraPerObject.requireExchange(name);
		if (shiftDataType != 0) lov->extraPerObject.requireShift(name, shiftDataType);
	}
};

#include "views/ov.h"




