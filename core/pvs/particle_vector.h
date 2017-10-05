#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/cuda_common.h>

#include <map>

#if __cplusplus < 201400L
namespace std
{
	template<typename T, typename... Args>
	std::unique_ptr<T> make_unique(Args&&... args)
	{
		return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}
}
#endif

class LocalParticleVector
{
protected:
	int np;

	// Store any additional data
	using DataMap = std::map<std::string, std::unique_ptr<GPUcontainer>>;
	DataMap dataPerParticle;

public:
	int changedStamp = 0;

	PinnedBuffer<Particle> coosvels;
	DeviceBuffer<Force> forces;


	// Local coordinate system; (0,0,0) is center of the local domain
	LocalParticleVector(int n=0, cudaStream_t stream = 0)
	{
		resize(n, stream);
	}

	int size()
	{
		return np;
	}

	virtual void resize(const int n, cudaStream_t stream)
	{
		assert(n>=0);

		coosvels.resize(n, stream);
		forces  .resize(n, stream);

		for (auto& kv : dataPerParticle)
			kv.second->resize(n, stream);

		np = n;
	}

	virtual void resize_anew(const int n)
	{
		assert(n>=0);

		coosvels.resize_anew(n);
		forces  .resize_anew(n);

		for (auto& kv : dataPerParticle)
			kv.second->resize_anew(n);

		np = n;
	}

	template<typename T>
	PinnedBuffer<T>* getDataPerParticle(const std::string& name)
	{
		GPUcontainer *contPtr;
		auto it = dataPerParticle.find(name);
		if (it == dataPerParticle.end())
		{
			warn("Requested extra data entry PER PARTICLE '%s' was absent, creating now", name.c_str());

			auto ptr = std::make_unique< PinnedBuffer<T> >(size());
			contPtr = ptr.get();
			dataPerParticle[name] = std::move(ptr);
		}
		else
		{
			contPtr = it->second.get();
		}

		return dynamic_cast< PinnedBuffer<T>* > (contPtr);
	}

	const DataMap& getDataPerParticleMap() const
	{
		return dataPerParticle;
	}

	virtual ~LocalParticleVector() = default;
};

class ParticleVector
{
public:
	float3 localDomainSize, globalDomainStart;
	LocalParticleVector *_local, *_halo;

	float mass;
	std::string name;
	// Local coordinate system; (0,0,0) is center of the local domain

protected:
	ParticleVector(	std::string name, float mass, LocalParticleVector *local, LocalParticleVector *halo ) :
		name(name), mass(mass), _local(local), _halo(halo) {}

public:
	ParticleVector(std::string name, float mass, int n=0) :
		name(name), mass(mass),
		_local( new LocalParticleVector(n) ),
		_halo ( new LocalParticleVector(0) )
	{}

	LocalParticleVector* local() { return _local; }
	LocalParticleVector* halo()  { return _halo;  }

	virtual void checkpoint(MPI_Comm comm, std::string path);
	virtual void restart(MPI_Comm comm, std::string path);

	virtual ~ParticleVector() { delete _local; delete _halo; }
};

#include "views/pv.h"


