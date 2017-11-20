#pragma once

#include "particle_vector.h"

#include <core/containers.h>
#include <core/logger.h>
#include <core/mesh.h>

class LocalObjectVector: public LocalParticleVector
{
protected:
	int objSize  = 0;
	DataMap dataPerObject;

public:
	int nObjects = 0;

	bool comExtentValid = false;

	// Helper buffers, used when a view with extra data is created
	PinnedBuffer<int> extraDataSizes;
	PinnedBuffer<char*> extraDataPtrs;
	PinnedBuffer<int> shiftingDataOffsets;

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


	void resize(const int np, cudaStream_t stream) override
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;
		LocalParticleVector::resize(np, stream);

		for (auto& kv : dataPerObject)
			kv.second->resize(nObjects, stream);
	}

	void resize_anew(const int np) override
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

			if (sizeof(T) % 4 != 0)
				die("Size of extra data per object must be a multiplier of 4 bytes");

			auto ptr = std::make_unique< PinnedBuffer<T> >(nObjects);
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

	bool checkDataPerObject(const std::string& name) const
	{
		return dataPerObject.find(name) != dataPerObject.end();
	}

	DataMap& getDataPerObjectMap()
	{
		return dataPerObject;
	}

	const DataMap& getDataPerObjectMap() const
	{
		return dataPerObject;
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




