#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/domain.h>

#include <vector>

class ParticleVector;
class CellList;

class Average3D : public SimulationPlugin
{
public:
    enum class ChannelType : int
    {
        Scalar, Vector_float3, Vector_float4, Vector_2xfloat4, Tensor6
    };

    struct HostChannelsInfo
    {
        int n;
        std::vector<std::string> names;
        PinnedBuffer<ChannelType> types;
        PinnedBuffer<float*> averagePtrs, dataPtrs;
        std::vector<DeviceBuffer<float>> average;
    };

    Average3D(std::string name, const YmrState *state,
              std::vector<std::string> pvNames,
              std::vector<std::string> channelNames, std::vector<Average3D::ChannelType> channelTypes,
              int sampleEvery, int dumpEvery, float3 binSize);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

protected:
    std::vector<std::string> pvNames;
    int nSamples;
    int sampleEvery, dumpEvery;
    int3 resolution;
    float3 binSize;

    DeviceBuffer<float>   density;
    PinnedBuffer<double>  accumulated_density;
    std::vector<char> sendBuffer;

    std::vector<ParticleVector*> pvs;

    HostChannelsInfo channelsInfo;
    std::vector<PinnedBuffer<double>> accumulated_average;
    
    DomainInfo domain;

    int getNcomponents(ChannelType type) const;
    void accumulateSampledAndClear(cudaStream_t stream);
    void scaleSampled(cudaStream_t stream);

    void sampleOnePv(ParticleVector *pv, cudaStream_t stream);
};

