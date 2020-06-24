// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <memory>
#include <string>

namespace mirheo
{

class ParticleVector;
class ParticlePacker;

class ExchangePVSFluxPlanePlugin : public SimulationPlugin
{
public:
    ExchangePVSFluxPlanePlugin(const MirState *state, std::string name, std::string pv1Name, std::string pv2Name, real4 plane);
    ~ExchangePVSFluxPlanePlugin();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pv1Name_, pv2Name_;
    ParticleVector *pv1_, *pv2_;
    real4 plane_;

    PinnedBuffer<int> numberCrossedParticles_;
    std::unique_ptr<ParticlePacker> extra1_, extra2_;
};

} // namespace mirheo
