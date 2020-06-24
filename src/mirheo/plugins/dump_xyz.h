// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class CellList;

class XYZPlugin : public SimulationPlugin
{
public:
    XYZPlugin(const MirState *state, std::string name, std::string pvNames, int dumpEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    int dumpEvery_;
    std::vector<char> sendBuffer_;
    ParticleVector *pv_;
    HostBuffer<real4> positions_;
};


class XYZDumper : public PostprocessPlugin
{
public:
    XYZDumper(std::string name, std::string path);
    ~XYZDumper();

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string path_;
    bool activated_ {true};
    std::vector<real4> pos_;
};

} // namespace mirheo
