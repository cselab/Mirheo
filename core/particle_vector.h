#pragma once

#include <string>
#include <core/datatypes.h>

class CellList;

struct ParticleVector
{
	int np;
	float mass;
	std::string name;

	PinnedBuffer<Particle> coosvels;//, pingPongCoosvels;
	//PinnedBuffer<Force> forces, pingPongForces;
	DeviceBuffer<Force> forces, pingPongForces;

	// Local coordinate system, (0,0,0) is center of the local domain
	float3 domainLength;

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
		//pingPongCoosvels.pushStream(stream);
		forces.pushStream(stream);
		pingPongForces.pushStream(stream);
	}

	virtual void popStreamWOhalo()
	{
		coosvels.popStream();
		//pingPongCoosvels.popStream();
		forces.popStream();
		pingPongForces.popStream();
	}

	virtual void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve)
	{
		assert(n>=0);

		coosvels.resize(n, kind);
		//pingPongCoosvels.resize(n, kind);
		forces.resize(n, kind);
		pingPongForces.resize(n, kind);

		np = n;
	}

	virtual ~ParticleVector() = default;
};

