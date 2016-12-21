#pragma once

class Simulation;

class Plugin
{
protected:

	Simulation* sim;

public:

	Plugin(Simulation* sim) : sim(sim) {};

	virtual void beforeForces(cudaStream_t stream) {};
	virtual void beforeIntegration(cudaStream_t stream) {};
	virtual void afterIntegration(cudaStream_t stream) {};

	virtual void dumpingSerialize(cudaStream_t stream) {};
	virtual void dumpingDeserializeAndDump() {};

	virtual ~Plugin() {};
};
