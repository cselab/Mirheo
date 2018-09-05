#pragma once

#include <vector>
#include <string>

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <plugins/write_xdmf_particles.h>

class ParticleVector;
class CellList;

class ParticleSenderPlugin : public SimulationPlugin
{
public:

    enum class ChannelType {
        Scalar, Vector, Tensor6
    };
    
    ParticleSenderPlugin(std::string name, std::string pvName, int dumpEvery,
                         std::vector<std::string> channelNames,
                         std::vector<ChannelType> channelTypes);

    void setup(Simulation *sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }
    
private:
    std::string pvName;
    ParticleVector *pv;
    
    int dumpEvery;

    HostBuffer<Particle> particles;
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

private:
    std::unique_ptr<XDMFParticlesDumper> dumper;
    std::string path;

    std::vector<Particle> particles;
    std::vector<float> positions, velocities;

    std::vector<std::string> channelNames;
    std::vector<std::vector<float>> channelData;

};
