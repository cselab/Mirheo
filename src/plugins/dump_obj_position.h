#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

class ObjectVector;

class ObjPositionsPlugin : public SimulationPlugin
{
public:
    ObjPositionsPlugin(std::string name, std::string ovName, int dumpEvery);

    void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string ovName;
    int dumpEvery;

    std::vector<char> sendBuffer;

    ObjectVector* ov;
};


class ObjPositionsDumper : public PostprocessPlugin
{
public:
    ObjPositionsDumper(std::string name, std::string path);

    void deserialize(MPI_Status& stat) override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

    ~ObjPositionsDumper() {};

private:
    std::string path;
    int3 nranks3D;

    bool activated = true;
    MPI_File fout;
};
