#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;

class Average3D : public SimulationPlugin
{
public:
    enum class ChannelType : int
    {
     Scalar, Vector_real3, Vector_real4, Tensor6, None
    };

    struct HostChannelsInfo
    {
        int n;
        std::vector<std::string> names;
        PinnedBuffer<ChannelType> types;
        PinnedBuffer<real*> averagePtrs, dataPtrs;
        std::vector<DeviceBuffer<real>> average;
    };

    Average3D(const MirState *state, std::string name,
              std::vector<std::string> pvNames, std::vector<std::string> channelNames,
              int sampleEvery, int dumpEvery, real3 binSize);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }


private:
    static const std::string numberDensityChannelName_;

    std::vector<std::string> pvNames_;

protected:
    
    std::vector<ParticleVector*> pvs_;

    int nSamples_ {0};
    int sampleEvery_, dumpEvery_;
    int3 resolution_;
    real3 binSize_;
    int3 rank3D_, nranks3D_;

    DeviceBuffer<real>   numberDensity_;
    PinnedBuffer<double> accumulatedNumberDensity_;

    HostChannelsInfo channelsInfo_;
    std::vector<PinnedBuffer<double>> accumulatedAverage_;

    std::vector<char> sendBuffer_;

    int getNcomponents(ChannelType type) const;
    void accumulateSampledAndClear(cudaStream_t stream);
    void scaleSampled(cudaStream_t stream);

    void sampleOnePv(ParticleVector *pv, cudaStream_t stream);
};

} // namespace mirheo
