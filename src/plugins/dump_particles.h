#pragma once

#include <vector>
#include <string>

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>

#include <core/xdmf/xdmf.h>

class ParticleVector;
class CellList;

class ParticleSenderPlugin : public SimulationPlugin
{
public:

    enum class ChannelType {
        Scalar, Vector, Tensor6
    };
    
    ParticleSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                         std::vector<std::string> channelNames,
                         std::vector<ChannelType> channelTypes);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }
    
protected:
    std::string pvName;
    ParticleVector *pv;
    
    int dumpEvery;

    HostBuffer<float4> positions, velocities;
    std::vector<std::string> channelNames;
    std::vector<ChannelType> channelTypes;
    std::vector<HostBuffer<float>> channelData;

    std::vector<char> sendBuffer;
};


class ParticleDumperPlugin : public PostprocessPlugin
{
public:
    ParticleDumperPlugin(std::string name, std::string path);

    void deserialize(MPI_Status& stat) override;
    void handshake() override;

protected:

    void _recvAndUnpack(MirState::TimeType &time, MirState::StepType& timeStamp);
    
    static constexpr int zeroPadding = 5;
    std::string path;

    std::vector<float4> pos4, vel4;
    std::vector<float3> velocities;
    std::vector<int64_t> ids;
    std::shared_ptr<std::vector<float3>> positions;

    std::vector<XDMF::Channel> channels;
    std::vector<std::vector<float>> channelData;
};
