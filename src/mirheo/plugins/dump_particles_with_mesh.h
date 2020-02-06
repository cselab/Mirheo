#pragma once

#include "dump_particles.h"

namespace mirheo
{

class ParticleWithMeshSenderPlugin : public ParticleSenderPlugin
{
public:

    ParticleWithMeshSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                                 const std::vector<std::string>& channelNames);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;
};


class ParticleWithMeshDumperPlugin : public ParticleDumperPlugin
{
public:
    ParticleWithMeshDumperPlugin(std::string name, std::string path);

    void handshake() override;
    void deserialize() override;

private:
    void _prepareConnectivity(int totNVertices);

private:
    std::shared_ptr<std::vector<int3>> allTriangles_;
    
    int nvertices_;
    std::vector<int3> triangles_;
};

} // namespace mirheo
