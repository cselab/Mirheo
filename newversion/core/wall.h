#pragma once

#include "datatypes.h"
#include "containers.h"
#include "logger.h"
#include "iniparser.h"

#include <mpi.h>
#include <string>

class Wall
{
private:

	MPI_Comm wallComm;

	std::vector<ParticleVector*> particleVectors;

	DeviceBuffer<Particle> frozen;
	cudaTextureObject_t sdfTex;
	cudaArray *sdfArray;
	DeviceBuffer<float> sdfRawData;
	DeviceBuffer<int> boundaryCells;

	float3 length, h;
	int3 resolution;

	IniParser& config;

	float dt;
	// TODO:
	const float rc = 1.0f;

public:

	Wall(MPI_Comm& comm, IniParser& config);

	void create(ParticleVector& dpds);
	void attach(ParticleVector* pv);
	void computeInteractions(cudaStream_t stream = 0);
	void bounce(cudaStream_t stream = 0);
};
