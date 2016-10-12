#pragma once

#include "datatypes.h"

struct ParticleVector
{
	int np;

	PinnedBuffer<Particle>     coosvels, pingPongBuf;
	DeviceBuffer<Acceleration> accs;

	DeviceBuffer<int> cellsStart;
	DeviceBuffer<uint8_t> cellsSize;
	float3 domainStart, length; // assume 0,0,0 is center of the local domain
	int3 ncells;
	int totcells;

	PinnedBuffer<Particle>	   halo;

	ParticleVector(int3 ncells, float3 domainStart, float3 length) : ncells(ncells), domainStart(domainStart), length(length)
	{
		int maxdim = std::max({ncells.x, ncells.y, ncells.z});
		int minpow2 = 1;
		while (minpow2 < maxdim) minpow2 *= 2;
		totcells = minpow2*minpow2*minpow2;

		cellsStart.resize(totcells + 1);
		cellsSize.resize(totcells + 1);
	}

	void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve, cudaStream_t stream = 0)
	{
		// TODO: stream
		coosvels.resize(n, kind, stream);
		pingPongBuf.resize(n, kind, stream);
		accs.resize(n, kind, stream);

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
