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

	ParticleVector(float3 domainStart, float3 domainLength, cudaStream_t stream) :
		domainStart(domainStart), domainLength(domainLength), received(0),
		coosvels(0, stream), pingPongBuf(0, stream), forces(0, stream)
	{
	}

	void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve)
	{
		coosvels.resize(n, kind);
		pingPongBuf.resize(n, kind);
		forces.resize(n, kind);

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
