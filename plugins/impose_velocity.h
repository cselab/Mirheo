#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <vector>
#include <string>

#include "utils.h"

class ParticleVector;

class ImposeVelocityPlugin : public SimulationPlugin
{
public:
	ImposeVelocityPlugin(std::string name, std::string pvName, float3 low, float3 high, float3 targetVel, int every) :
		SimulationPlugin(name), pvName(pvName), low(low), high(high), targetVel(targetVel), every(every)
	{	}

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
	void afterIntegration(cudaStream_t stream) override;

	bool needPostproc() override { return false; }

	~ImposeVelocityPlugin() = default;

private:
	std::string pvName;
	ParticleVector* pv;

	float3 high, low;
	float3 targetVel;

	int every;

	PinnedBuffer<int> nSamples{1};
	PinnedBuffer<float3> totVel{1};
};

