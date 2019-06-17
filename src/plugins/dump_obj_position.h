#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

#include <core/pvs/object_vector.h>
#include <core/rigid_kernels/rigid_motion.h>


class ObjStatsPlugin : public SimulationPlugin
{
public:
    ObjStatsPlugin(const YmrState *state, std::string name, std::string ovName, int dumpEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string ovName;
    int dumpEvery;
    bool needToSend = false;
    
    HostBuffer<int64_t> ids;
    HostBuffer<COMandExtent> coms;
    HostBuffer<RigidMotion> motions;
    YmrState::TimeType savedTime = 0;

    std::vector<char> sendBuffer;

    ObjectVector *ov;
};


class ObjStatsDumper : public PostprocessPlugin
{
public:
    ObjStatsDumper(std::string name, std::string path);
    ~ObjStatsDumper();

    void deserialize(MPI_Status& stat) override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path;
    int3 nranks3D;

    bool activated = true;
    MPI_File fout;
};
