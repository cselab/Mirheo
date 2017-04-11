#pragma once

#include <string>
#include <core/datatypes.h>

class CellList;

struct ParticleVector
{
	int np;
	float mass;
	std::string name;

	PinnedBuffer<Particle> coosvels;
	DeviceBuffer<Force> forces;

	// Local coordinate system, (0,0,0) is center of the local domain
	float3 domainSize, globalDomainStart;

	PinnedBuffer<Particle> halo;
	int changedStamp;

	ParticleVector(std::string name, int n=0) : name(name), changedStamp(0)
	{
		resize(n);
	}

	int size()
	{
		return np;
	}

	virtual void pushStreamWOhalo(cudaStream_t stream)
	{
		coosvels.pushStream(stream);
		forces.pushStream(stream);
	}

	virtual void popStreamWOhalo()
	{
		coosvels.popStream();
		forces.popStream();
	}

	virtual void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve)
	{
		assert(n>=0);

		coosvels.resize(n, kind);
		forces.resize(n, kind);

		np = n;
	}

	virtual ~ParticleVector() = default;
};

