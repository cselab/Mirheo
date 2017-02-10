#pragma once

#include <plugins/plugin.h>
#include <core/datatypes.h>

#include <vector>

class ParticleVector;
class CellList;

class SimulationStats : public SimulationPlugin
{
private:
	int fetchEvery;
	int invoked = 0;

	std::vector<Particle> allParticles;
	HostBuffer<char> sendBuffer;

public:
	SimulationStats(std::string name, int fetchEvery) :
		SimulationPlugin(name), fetchEvery(fetchEvery) {}

	void handshake();
	void afterIntegration(float t);
	void serializeAndSend();

	~SimulationStats() {};
};

class PostprocessStats : public PostprocessPlugin
{
private:
	std::vector<Particle> coosvels;

public:
	PostprocessStats(std::string name) : PostprocessPlugin(name) { }

	void deserialize(MPI_Status& stat);
	void handshake();

	~PostprocessStats() {};
};
