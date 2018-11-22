#pragma once

#include <plugins/interface.h>
#include <string>

#include <core/utils/folders.h>

class RigidObjectVector;

class MagneticOrientationPlugin : public SimulationPlugin
{
public:
    MagneticOrientationPlugin(std::string name, std::string rovName, float3 moment);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rovName;
    RigidObjectVector* rov;
    float3 moment;
};

