// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <memory>
#include <string>

namespace mirheo
{

namespace rdf_plugin
{
using ReductionType = double;
using CountType = unsigned long long;
}

class CellList;
class ParticleVector;

class RdfPlugin : public SimulationPlugin
{
public:
    RdfPlugin(const MirState *state, std::string name, std::string pvName, real maxDist, int nbins, int computeEvery);
    ~RdfPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    real maxDist_;
    int nbins_;
    int computeEvery_;

    bool needToDump_{false};
    ParticleVector *pv_;

    PinnedBuffer<rdf_plugin::CountType> nparticles_{1};
    PinnedBuffer<rdf_plugin::CountType> countsPerBin_;

    std::vector<char> sendBuffer_;
    std::unique_ptr<CellList> cl_;
};


class RdfDump : public PostprocessPlugin
{
public:
    RdfDump(std::string name, std::string basename);

    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void deserialize() override;

private:
    std::string basename_;
};

} // namespace mirheo
