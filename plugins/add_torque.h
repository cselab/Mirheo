#pragma once

#include <plugins/interface.h>
#include <vector>
#include <string>

#include <core/utils/folders.h>

class RigidObjectVector;

class AddTorquePlugin : public SimulationPlugin
{
public:
	AddTorquePlugin(std::string name, std::string rovName, float3 torque) :
		SimulationPlugin(name), rovName(rovName), torque(torque)
	{	}

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
	void beforeForces(cudaStream_t stream) override;

	bool needPostproc() override { return false; }

	~AddTorquePlugin() = default;

private:
	std::string rovName;
	RigidObjectVector* rov;
	float3 torque;
};

