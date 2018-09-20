#pragma once

#include <plugins/interface.h>
#include <plugins/channel_dumper.h>
#include <plugins/average_flow.h>
#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

class ParticleVector;
class ObjectVector;
class CellList;

class AverageRelative3D : public Average3D
{
public:
    AverageRelative3D(
            std::string name,
            std::vector<std::string> pvNames,
            std::vector<std::string> channelNames, std::vector<Average3D::ChannelType> channelTypes,
            int sampleEvery, int dumpEvery, float3 binSize,
            std::string relativeOVname, int relativeID);

    void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    ObjectVector* relativeOV{nullptr};
    std::string relativeOVname;
    int relativeID;

    float3 averageRelativeVelocity{0, 0, 0};

    int3 localResolution;

    std::vector<std::vector<double>> localChannels;
    std::vector<double> localDensity;

    void extractLocalBlock();

    void sampleOnePv(float3 relativeParam, ParticleVector *pv, cudaStream_t stream);
};


