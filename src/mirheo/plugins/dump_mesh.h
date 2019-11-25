#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class ObjectVector;
class CellList;

class MeshPlugin : public SimulationPlugin
{
private:
    std::string ovName;
    int dumpEvery;

    std::vector<char> sendBuffer;
    std::vector<real3> vertices;
    PinnedBuffer<real4>* srcVerts;

    ObjectVector* ov;

public:
    MeshPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery);

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

    std::vector<int3> connectivity;
    std::vector<real3> vertices;

public:
    MeshDumper(std::string name, std::string path);
    ~MeshDumper();
    
    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
};

} // namespace mirheo
