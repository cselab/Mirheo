#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>
#include <mirheo/plugins/average_flow.h>
#include <mirheo/plugins/channel_dumper.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class ObjectVector;
class CellList;

class AverageRelative3D : public Average3D
{
public:
    AverageRelative3D(const MirState *state, std::string name,
                      std::vector<std::string> pvNames,
                      std::vector<std::string> channelNames,
                      int sampleEvery, int dumpEvery, real3 binSize,
                      std::string relativeOVname, int relativeID);

  void setup(Simulation *simulation, const MPI_Comm &comm,
             const MPI_Comm &interComm) override;
  void afterIntegration(cudaStream_t stream) override;
  void serializeAndSend(cudaStream_t stream) override;

  bool needPostproc() override { return true; }

private:
    ObjectVector *relativeOV_ {nullptr};
    std::string relativeOVname_;
    int relativeID_;

    real3 averageRelativeVelocity_ {0, 0, 0};
    int3 localResolution_;

    std::vector<std::vector<double>> localChannels_;
    std::vector<double> localNumberDensity_;

    void extractLocalBlock();

    void sampleOnePv(real3 relativeParam, ParticleVector *pv, cudaStream_t stream);
};

} // namespace mirheo
