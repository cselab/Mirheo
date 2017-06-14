#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/cuda_common.h>

class LocalParticleVector
{
protected:
	int np;

public:
	int changedStamp = 0;

	PinnedBuffer<Particle> coosvels;
	DeviceBuffer<Force> forces;

	// Local coordinate system; (0,0,0) is center of the local domain

	LocalParticleVector(int n=0)
	{
		resize(n);
	}

	int size()
	{
		return np;
	}

	virtual void pushStream(cudaStream_t stream)
	{
		coosvels.pushStream(stream);
		forces  .pushStream(stream);
	}

	virtual void popStream()
	{
		coosvels.popStream();
		forces  .popStream();
	}

	virtual void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve)
	{
		assert(n>=0);

		coosvels.resize(n, kind);
		forces  .resize(n, kind);

		np = n;
	}

	virtual ~LocalParticleVector() = default;
};

class ParticleVector
{
public:
	LocalParticleVector *_local, *_halo;

	float mass;
	std::string name;
	// Local coordinate system; (0,0,0) is center of the local domain
	float3 domainSize, globalDomainStart;

protected:
	ParticleVector(	std::string name, LocalParticleVector *local, LocalParticleVector *halo ) :
		name(name), _local(local), _halo(halo) {}

public:
	ParticleVector(std::string name, int n=0) :
		name(name),
		_local( new LocalParticleVector(n) ),
		_halo ( new LocalParticleVector(n) )
	{}

	LocalParticleVector* local() { return _local; }
	LocalParticleVector* halo()  { return _halo;  }

	__forceinline__ __host__ __device__ float3 local2global(float3 x)
	{
		return x + globalDomainStart + 0.5f * domainSize;
	}

	virtual ~ParticleVector() { delete _local; delete _halo; }
};
