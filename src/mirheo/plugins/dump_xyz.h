#pragma once

#include <mirheo/plugins/interface.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>

#include <vector>

class ParticleVector;
class CellList;

class XYZPlugin : public SimulationPlugin
{
private:
    std::string pvName;
    int dumpEvery;

    std::vector<char> sendBuffer;

    ParticleVector* pv;
    
    HostBuffer<real4> positions;

public:
    XYZPlugin(const MirState *state, std::string name, std::string pvNames, int dumpEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }
};


class XYZDumper : public PostprocessPlugin
{
private:
    std::string path;
    int3 nranks3D;

    bool activated = true;

    std::vector<real4> pos;

public:
    XYZDumper(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;

    ~XYZDumper() {};
};
