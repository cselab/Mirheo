#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <plugins/write_xdmf.h>

#include <vector>

class ParticleVector;
class CellList;

class Avg3DPlugin : public SimulationPlugin
{
private:
	std::string pvNames;
	int nSamples;
	int sampleEvery, dumpEvery;
	int3 resolution;
	float3 binSize;
	bool needDensity, needMomentum, needForce;

	PinnedBuffer<float>  density;
	PinnedBuffer<float3> momentum, force;
	std::vector<char> sendBuffer;

	std::vector<ParticleVector*> particleVectors;

public:
	Avg3DPlugin(std::string name, std::string pvNames, int sampleEvery, int dumpEvery, float3 binSize,
			bool needMomentum, bool needForce);

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm);
	void handshake();
	void afterIntegration(cudaStream_t stream);
	void serializeAndSend(cudaStream_t stream);

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

	std::vector<float>  density;
	std::vector<float3> momentum, force;

public:
	Avg3DDumper(std::string name, std::string path);

	void deserialize(MPI_Status& stat);
	void handshake();

	~Avg3DDumper() {};
};
