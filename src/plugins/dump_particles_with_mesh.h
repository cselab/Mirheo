#pragma once

#include "dump_particles.h"

class ParticleWithMeshSenderPlugin : public ParticleSenderPlugin
{
public:

    ParticleWithMeshSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                                 std::vector<std::string> channelNames,
                                 std::vector<ChannelType> channelTypes);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;
};


class ParticleWithMeshDumperPlugin : public ParticleDumperPlugin
{
public:
    ParticleWithMeshDumperPlugin(std::string name, std::string path);

    void handshake() override;
    void deserialize(MPI_Status& stat) override;

protected:
    std::shared_ptr<std::vector<int3>> allTriangles;

    void _prepareConnectivity(int totNVertices);
    
    int nvertices;
    std::vector<int3> triangles;
};
