#pragma once

#include <core/containers.h>
#include "particle_vector.h"

#include <core/logger.h>

struct ObjectMesh
{
	static const int maxDegree = 7;
	int nvertices, ntriangles;

	PinnedBuffer<int3> triangles;
	PinnedBuffer<int> adjacent, adjacent_second;
};


class LocalObjectVector: public LocalParticleVector
{
protected:
	int objSize  = 0;
	DataMap dataPerObject;

public:
	int nObjects = 0;

	// Helper buffers, used when a view is created
	PinnedBuffer<int> extraDataSizes;
	PinnedBuffer<char*> extraDataPtrs;

	struct __align__(16) COMandExtent
	{
		float3 com, low, high;
	};


	LocalObjectVector(const int objSize, const int nObjects = 0, cudaStream_t stream = 0) :
		LocalParticleVector(objSize*nObjects), objSize(objSize), nObjects(nObjects)
	{
		resize(nObjects*objSize, stream);
		dataPerObject["com_extents"] = std::make_unique< PinnedBuffer<COMandExtent> >(nObjects);
		dataPerObject["ids"] = std::make_unique< PinnedBuffer<int> >(nObjects);
	};


	virtual void resize(const int np, cudaStream_t stream)
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;
		LocalParticleVector::resize(np, stream);

		for (auto& kv : dataPerObject)
			kv.second->resize(nObjects, stream);
	}

	virtual void resize_anew(const int np)
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;
		LocalParticleVector::resize_anew(np);

		for (auto& kv : dataPerObject)
			kv.second->resize_anew(nObjects);
	}

	template<typename T>
	PinnedBuffer<T>* getDataPerObject(const std::string& name)
	{
		GPUcontainer *contPtr;
		auto it = dataPerObject.find(name);
		if (it == dataPerObject.end())
		{
			warn("Requested extra data entry PER OBJECT '%s' was absent, creating now", name.c_str());

			auto ptr = std::make_unique< PinnedBuffer<T> >(size());
			contPtr = ptr.get();
			dataPerObject[name] = std::move(ptr);
		}
		else
		{
			contPtr = it->second.get();
		}

		auto res = dynamic_cast< PinnedBuffer<T>* > (contPtr);
		if (res == nullptr)
			error("Wrong type of particle extra data entry '%s'", name.c_str());

		return res;
	}

	const DataMap& getDataPerObjectMap() const
	{
		return dataPerObject;
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
	ObjectMesh mesh;

	ObjectVector(std::string name, float mass, const int objSize, const int nObjects = 0) :
		ObjectVector( name, mass, objSize,
					  new LocalObjectVector(objSize, nObjects),
					  new LocalObjectVector(objSize, 0) )
	{}

	virtual void getMeshWithVertices(ObjectMesh* mesh, PinnedBuffer<Particle>* vertices, cudaStream_t stream);
	virtual void findExtentAndCOM(cudaStream_t stream);

	LocalObjectVector* local() { return static_cast<LocalObjectVector*>(_local); }
	LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(_halo);  }

	virtual ~ObjectVector() = default;
};

#include "views/ov.h"



