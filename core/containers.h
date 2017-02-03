#pragma once

#include <string>
#include "datatypes.h"

struct ParticleVector
{
	int np;
	float mass;
	std::string name;

	PinnedBuffer<Particle>     coosvels, pingPongBuf;
	DeviceBuffer<Force> forces;

	float3 domainStart, domainLength; // assume 0,0,0 is center of the local domain
	int received;

	PinnedBuffer<Particle>	   halo;

	ParticleVector(std::string name) : name(name) {}

	void setStream(cudaStream_t stream)
	{
		coosvels.setStream(stream);
		pingPongBuf.setStream(stream);
		forces.setStream(stream);
		halo.setStream(stream);
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
