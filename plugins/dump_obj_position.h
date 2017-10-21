#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

class ObjectVector;

class ObjPositionsPlugin : public SimulationPlugin
{
private:
	std::string ovName;
	int dumpEvery;

	ObjectVector* ov;

public:
	ObjPositionsPlugin(std::string name, std::string ovName, int dumpEvery);

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm);

	void beforeForces(cudaStream_t stream) override;
	void serializeAndSend(cudaStream_t stream) override;

	~ObjPositionsPlugin() {};
};


class ObjPositionsDumper : public PostprocessPlugin
{
private:
	std::string path;
	int3 nranks3D;

	bool activated = true;
	int timeStamp = 0;

public:
	ObjPositionsDumper(std::string name, std::string path);

	void deserialize(MPI_Status& stat) override;
	void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;

	~ObjPositionsDumper() {};
};
