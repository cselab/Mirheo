#pragma once

#include <plugins/plugin.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <plugins/timer.h>

#include <vector>

class ParticleVector;
class CellList;

using ReductionType = float;

class SimulationStats : public SimulationPlugin
{
private:
	int fetchEvery;
	bool needToDump;

	int nparticles;
	PinnedBuffer<ReductionType> momentum, energy;
	std::vector<char> sendBuffer;

	Timer<> timer;

public:
	SimulationStats(std::string name, int fetchEvery) :
		SimulationPlugin(name), fetchEvery(fetchEvery), needToDump(false), momentum(3), energy(1)
	{
		timer.start();
	}

	void afterIntegration(cudaStream_t stream);
	void serializeAndSend(cudaStream_t stream);

	~SimulationStats() {};
};

class PostprocessStats : public PostprocessPlugin
{
private:
	std::vector<Particle> coosvels;
	MPI_Datatype mpiReductionType;

public:
	PostprocessStats(std::string name);

	void deserialize(MPI_Status& stat);

	~PostprocessStats() {};
};
