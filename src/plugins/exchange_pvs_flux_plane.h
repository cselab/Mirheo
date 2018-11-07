#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <vector>
#include <string>

class ParticleVector;

class ExchangePVSFluxPlanePlugin : public SimulationPlugin
{
public:
    ExchangePVSFluxPlanePlugin(std::string name, std::string pv1Name, std::string pv2Name, float4 plane);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeParticleDistribution(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pv1Name, pv2Name;
    ParticleVector *pv1, *pv2;
    float4 plane;

    PinnedBuffer<int> numberCrossedParticles;
};

