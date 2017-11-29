#pragma once

#include <plugins/interface.h>
#include <vector>
#include <string>

#include "utils.h"

class ParticleVector;

class TemperaturizePlugin : public SimulationPlugin
{
public:
	TemperaturizePlugin(std::string name, std::string pvName, float kbT, bool keepVelocity) :
		SimulationPlugin(name), pvName(pvName), kbT(kbT), keepVelocity(keepVelocity)
	{	}

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
	void beforeForces(cudaStream_t stream) override;

	bool needPostproc() override { return false; }

	~TemperaturizePlugin() = default;

private:
	std::string pvName;
	ParticleVector* pv;
	float kbT;
	bool keepVelocity;
};

