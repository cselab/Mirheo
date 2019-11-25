#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <mirheo/core/xdmf/xdmf.h>

#include <vector>
#include <string>

namespace mirheo
{

class ParticleVector;
class CellList;

class ParticleSenderPlugin : public SimulationPlugin
{
public:
    ParticleSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                         const std::vector<std::string>& channelNames);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }
    
protected:
    std::string pvName;
    ParticleVector *pv;
    
    int dumpEvery;

    HostBuffer<real4> positions, velocities;
    std::vector<std::string> channelNames;
    DeviceBuffer<char> workSpace;
    std::vector<HostBuffer<char>> channelData;

    std::vector<char> sendBuffer;
};


class ParticleDumperPlugin : public PostprocessPlugin
{
public:
    ParticleDumperPlugin(std::string name, std::string path);

    void deserialize() override;
    void handshake() override;

protected:

    void _recvAndUnpack(MirState::TimeType &time, MirState::StepType& timeStamp);
    
    static constexpr int zeroPadding = 5;
    std::string path;

    std::vector<real4> pos4, vel4;
    std::vector<real3> velocities;
    std::vector<int64_t> ids;
    std::shared_ptr<std::vector<real3>> positions;

    std::vector<XDMF::Channel> channels;
    std::vector<std::vector<char>> channelData;
};

} // namespace mirheo
