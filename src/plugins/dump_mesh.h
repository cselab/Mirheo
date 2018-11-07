#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

class ParticleVector;
class ObjectVector;
class CellList;

class MeshPlugin : public SimulationPlugin
{
private:
    std::string ovName;
    int dumpEvery;

    std::vector<char> data;
    std::vector<float3> vertices;
    PinnedBuffer<Particle>* srcVerts;

    ObjectVector* ov;

public:
    MeshPlugin(std::string name, std::string ovName, int dumpEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }
};


class MeshDumper : public PostprocessPlugin
{
private:
    std::string path;

    bool activated = true;
    int timeStamp = 0;

    std::vector<int3> connectivity;
    std::vector<float3> vertices;

public:
    MeshDumper(std::string name, std::string path);

    void deserialize(MPI_Status& stat) override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;

    ~MeshDumper() {};
};
