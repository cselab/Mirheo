#pragma once

#include "plugin.h"
#include "../core/datatypes.h"
#include "../core/containers.h"
#include "../core/celllist.h"
#include "write_xdmf.h"

#include <vector>

class Avg3DPlugin : public SimulationPlugin
{
private:
	int nTimeSteps, nSamples;
	int sampleEvery, dumpEvery;
	int3 resolution;
	float3 h;
	bool needDensity, needMomentum, needForce;

	PinnedBuffer<float>  density;
	PinnedBuffer<float4> momentum, force;
	HostBuffer<char> sendBuffer;

	std::vector<std::pair<ParticleVector*, CellList*>> particlesAndCells;

public:
	Avg3DPlugin(int id, Simulation* sim, const MPI_Comm& comm, int sendRank,
			std::string pvNames, int sampleEvery, int dumpEvery, int3 resolution, float3 h,
			bool needDensity, bool needMomentum, bool needForce);

	void handshake();
	void afterIntegration(float t);
	void serializeAndSend();

	~Avg3DPlugin() {};
};


class Avg3DDumper : public PostprocessPlugin
{
private:
	XDMFDumper* dumper;

	int3 resolution;
	float3 h;
	bool needDensity, needMomentum, needForce;

	PinnedBuffer<float>  density;
	PinnedBuffer<float4> momentum, force;

public:
	Avg3DDumper(int id, MPI_Comm comm, int recvRank, std::string path);

	void deserialize() {};
	void handshake() {};

	~Avg3DDumper() {};
};
