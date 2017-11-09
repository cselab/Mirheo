#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <plugins/timer.h>

#include <vector>

class ParticleVector;
class CellList;

using ReductionType = double;

class SimulationStats : public SimulationPlugin
{
private:
	int fetchEvery;
	bool needToDump{false};

	int nparticles;
	PinnedBuffer<ReductionType> momentum{3}, energy{1};
	PinnedBuffer<float> maxvel{1};
	std::vector<char> sendBuffer;

	Timer<> timer;

public:
	SimulationStats(std::string name, int fetchEvery);

	void afterIntegration(cudaStream_t stream) override;
	void serializeAndSend(cudaStream_t stream) override;

	bool needPostproc() override { return true; }
};

class PostprocessStats : public PostprocessPlugin
{
private:
	std::vector<Particle> coosvels;
	MPI_Datatype mpiReductionType;

public:
	PostprocessStats(std::string name);

	void deserialize(MPI_Status& stat) override;
};
