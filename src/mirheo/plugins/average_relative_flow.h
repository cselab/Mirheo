// Copyright 2020 ETH Zurich. All Rights Reserved.
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

/** Perform the same task as AverageRelative3D on a grid that moves relatively to a given object's center of mass in a RigidObjectVector.

    Cannot be used with multiple invocations of `Mirheo.run`.
 */
class AverageRelative3D : public Average3D
{
public:
    /** Create an AverageRelative3D object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames The list of names of the ParticleVector that will be used when averaging.
        \param [in] channelNames The list of particle data channels to average. Will die if the channel does not exist.
        \param [in] sampleEvery Compute spatial averages every this number of time steps.
        \param [in] dumpEvery Compute time averages and send to the postprocess side every this number of time steps.
        \param [in] binSize Size of one spatial bin along the three axes.
        \param [in] relativeOVname Name of the RigidObjectVector that contains the reference object.
        \param [in] relativeID Index of the reference object within the RigidObjectVector.
     */
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
