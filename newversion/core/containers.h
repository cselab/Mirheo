#pragma once

#include "datatypes.h"

struct ParticleVector
{
	int np;

	PinnedBuffer<Particle>     coosvels, pingPongBuf;
	DeviceBuffer<Acceleration> accs;

	DeviceBuffer<int> cellsStart;
	DeviceBuffer<uint8_t> cellsSize;
	float3 domainStart; // assume 0,0,0 is center of the local domain
	int3 ncells;
	int totcells;

	PinnedBuffer<Particle>	   halo;

	ParticleVector(int3 ncells, float3 domainStart) : ncells(ncells), domainStart(domainStart)
	{
		totcells = ncells.x * ncells.y * ncells.z;
		cellsStart.resize(totcells + 1);
		cellsSize.resize(totcells + 1);
	}

	void resize(const int n, ResizeKind kind = ResizeKind::resizePreserve)
	{
		// TODO: stream
		coosvels.resize(n, kind);
		pingPongBuf.resize(n, kind);
		accs.resize(n, kind);

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
