#pragma once

#include <plugins/plugin.h>
#include <core/datatypes.h>
#include <plugins/timer.h>

#include <vector>

class ParticleVector;

class TemperaturizePlugin : public SimulationPlugin
{
private:
	std::vector<std::string> pvNames;
	std::vector<ParticleVector*> pvs;
	float kbT;

public:
	TemperaturizePlugin(std::string name, std::vector<std::string> pvNames, float kbT) :
		SimulationPlugin(name), pvNames(pvNames), kbT(kbT)
	{}

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm);
	void beforeForces(cudaStream_t stream);

	~TemperaturizePlugin() = default;
};

