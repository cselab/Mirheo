#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <plugins/write_xdmf.h>

#include <vector>

class ParticleVector;
class CellList;

class XYZPlugin : public SimulationPlugin
{
private:
	std::string pvName;
	int dumpEvery;

	PinnedBuffer<float>  density;
	PinnedBuffer<float3> momentum, force;
	std::vector<char> sendBuffer;

	ParticleVector* pv;

public:
	XYZPlugin(std::string name, std::string pvNames, int dumpEvery);

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm);

	void beforeForces(cudaStream_t stream) override;
	void serializeAndSend(cudaStream_t stream) override;

	~XYZPlugin() {};
};


class XYZDumper : public PostprocessPlugin
{
private:
	std::string path;
	int3 nranks3D;

	int timeStamp = 0;

public:
	XYZDumper(std::string name, std::string path);

	void deserialize(MPI_Status& stat);

	~XYZDumper() {};
};
