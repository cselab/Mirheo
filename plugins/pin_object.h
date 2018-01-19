#pragma once

#include <plugins/interface.h>
#include <vector>
#include <string>
#include <core/containers.h>

#include <core/utils/folders.h>

class ObjectVector;
class RigidObjectVector;

class PinObjectPlugin : public SimulationPlugin
{
public:
	PinObjectPlugin(std::string name, std::string ovName, int3 pinTranslation, int3 pinRotation, int reportEvery);

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
	void beforeIntegration(cudaStream_t stream) override;
	void afterIntegration(cudaStream_t stream) override;
	void serializeAndSend (cudaStream_t stream) override;
	void handshake() override;

	bool needPostproc() override { return true; }

	~PinObjectPlugin() = default;

private:
	std::string ovName;
	ObjectVector* ov;
	RigidObjectVector* rov{nullptr};

	int3 pinTranslation, pinRotation;

	int reportEvery;
	int count{0};

	PinnedBuffer<float4> forces, torques;
	std::vector<char> sendBuffer;
};

class ReportPinObjectPlugin : public PostprocessPlugin
{
public:
	ReportPinObjectPlugin(std::string name, std::string path);
	void deserialize(MPI_Status& stat) override;
	void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
	void handshake() override;

private:
	bool activated;
	std::string path;

	FILE* fout;
	std::vector<float4> forces, torques;
};
