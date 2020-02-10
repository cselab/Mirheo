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
public:
    MeshPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }
    void saveSnapshotAndRegister(Dumper& dumper) override;

protected:
    ConfigObject _saveSnapshot(Dumper& dumper, const std::string& typeName);

private:
    std::string ovName_;
    int dumpEvery_;

    std::vector<char> sendBuffer_;
    std::vector<real3> vertices_;
    PinnedBuffer<real4>* srcVerts_;

    ObjectVector *ov_;
};


class MeshDumper : public PostprocessPlugin
{
public:
    MeshDumper(std::string name, std::string path);
    ~MeshDumper();

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void saveSnapshotAndRegister(Dumper& dumper) override;

protected:
    ConfigObject _saveSnapshot(Dumper& dumper, const std::string& typeName);

private:
    std::string path_;

    bool activated_{true};

    std::vector<int3> connectivity_;
    std::vector<real3> vertices_;
};

} // namespace mirheo
