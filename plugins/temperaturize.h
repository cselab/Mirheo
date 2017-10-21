#pragma once

#include <plugins/interface.h>
#include <vector>
#include <string>

#include "utils.h"

class ParticleVector;

class TemperaturizePlugin : public SimulationPlugin
{
private:
	std::vector<std::string> pvNames;
	std::vector<ParticleVector*> pvs;
	float kbT;

public:
	TemperaturizePlugin(std::string name, std::string pvNames, float kbT) :
		SimulationPlugin(name), kbT(kbT)
	{
		this->pvNames = splitByDelim(pvNames);
	}

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm);
	void beforeForces(cudaStream_t stream);

	~TemperaturizePlugin() = default;
};

