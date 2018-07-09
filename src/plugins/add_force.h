#pragma once

#include <plugins/interface.h>
#include <vector>
#include <string>

#include <core/utils/folders.h>

class ParticleVector;

class AddForcePlugin : public SimulationPlugin
{
public:
    AddForcePlugin(std::string name, std::string pvName, float3 force) :
        SimulationPlugin(name), pvName(pvName), force(force)
    {    }

    void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    ~AddForcePlugin() = default;

private:
    std::string pvName;
    ParticleVector* pv;
    float3 force;
};

