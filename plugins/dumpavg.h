#pragma once
#include "plugin.h"
#include "../core/datatypes.h"
#include "../core/containers.h"
#include "../core/celllist.h"


#include <vector>

class DumpAvg3D : public Plugin
{
private:
	int sampleEvery;
	int3 resolution;
	bool needDensity, needVelocity, needForce;
	std::string namePrefix;

	PinnedBuffer<float>  density;
	PinnedBuffer<float4> velocity, force;

	CellList cellList;

	std::vector<ParticleVector*> particleVectors;

public:
	DumpAvg3D(Simulation* sim, std::string pvNames, int sampleEvery, int3 resolution, bool needDensity, bool needVelocity, bool needForce, std::string namePrefix);

	void afterIntegration(cudaStream_t stream);
};
