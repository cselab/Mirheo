#pragma once

#include <plugins/plugin.h>
#include <core/datatypes.h>
#include <plugins/write_xdmf.h>

#include <vector>

class ParticleVector;
class CellList;

class Avg3DPlugin : public SimulationPlugin
{
private:
	int nTimeSteps, nSamples;
	int sampleEvery, dumpEvery;
	int3 resolution;
	float3 h;
	bool needDensity, needMomentum, needForce;

	PinnedBuffer<float>  density;
	PinnedBuffer<float3> momentum, force;
	HostBuffer<char> sendBuffer;

	std::vector<std::pair<ParticleVector*, CellList*>> particlesAndCells;

public:
	Avg3DPlugin(std::string name, std::string pvNames, int sampleEvery, int dumpEvery, int3 resolution, float3 h,
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
	std::string path;

	int3 nranks3D;
	int3 resolution;
	float3 h;
	bool needDensity, needMomentum, needForce;

	PinnedBuffer<float>  density;
	PinnedBuffer<float3> momentum, force;

public:
	Avg3DDumper(std::string name, std::string path, int3 nranks3D);

	void deserialize(MPI_Status& stat);
	void handshake();

	~Avg3DDumper() {};
};
