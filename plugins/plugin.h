#pragma once

class Simulation;

class Plugin
{
protected:

	Simulation* sim;

public:

	virtual void attach(Simulation* sim)
	{
		this->sim = sim;
	}

	virtual void beforeForces() {};
	virtual void beforeIntegration() {};
	virtual void afterIntegration() {};

	virtual void dumpingSerialize() {};
	virtual void dumpingDeserializeAndDump() {};

	virtual ~Plugin() {};
};
