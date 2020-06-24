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
    std::string ovName_;
    int dumpEvery_;
    bool needToSend_ {false};

    HostBuffer<int64_t> ids_;
    HostBuffer<COMandExtent> coms_;
    HostBuffer<RigidMotion> motions_;
    DeviceBuffer<RigidMotion> motionStats_;
    HostBuffer<int> typeIds_;
    MirState::TimeType savedTime_ {0};
    bool isRov_ {false};
    bool hasTypeIds_ {false};

    std::vector<char> sendBuffer_;

    ObjectVector *ov_;
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
    std::string path_;
    int3 nranks3D_;

    bool activated_ = true;
    MPI_File fout_;
};

} // namespace mirheo
