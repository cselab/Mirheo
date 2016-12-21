#pragma once

#include "datatypes.h"

struct ParticleVector
{
	int np;
	float mass;

	PinnedBuffer<Particle>     coosvels, pingPongBuf;
	DeviceBuffer<Force> forces;

	float3 domainStart, domainLength; // assume 0,0,0 is center of the local domain
	int received;

	PinnedBuffer<Particle>	   halo;

	ParticleVector(float3 domainStart, float3 domainLength) : domainStart(domainStart), domainLength(domainLength), received(0)
	{
	}

	void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve, cudaStream_t stream = 0)
	{
		coosvels.resize(n, kind, stream);
		pingPongBuf.resize(n, kind, stream);
		forces.resize(n, kind, stream);

		np = n;
	}
};

class ObjectVector: public ParticleVector
{
	DeviceBuffer<int> objStarts;
};

class UniformObjectVector: public ObjectVector
{

};
