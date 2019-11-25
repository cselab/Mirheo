#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/pvs/object_vector.h>

#include <vector>

namespace mirheo
{

class ObjStatsPlugin : public SimulationPlugin
{
public:
    ObjStatsPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string ovName;
    int dumpEvery;
    bool needToSend {false};
    
    HostBuffer<int64_t> ids;
    HostBuffer<COMandExtent> coms;
    HostBuffer<RigidMotion> motions;
    DeviceBuffer<RigidMotion> motionStats;
    HostBuffer<int> typeIds;
    MirState::TimeType savedTime = 0;
    bool isRov {false};
    bool hasTypeIds {false};

    std::vector<char> sendBuffer;

    ObjectVector *ov;
};


class ObjStatsDumper : public PostprocessPlugin
{
public:
    ObjStatsDumper(std::string name, std::string path);
    ~ObjStatsDumper();

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path;
    int3 nranks3D;

    bool activated = true;
    MPI_File fout;
};

} // namespace mirheo
