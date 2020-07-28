// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>

#include <functional>
#include <vector>

namespace mirheo
{

class ParticleVector;

/** Functional used to constrain an array of positions or velocities over time.
    The input is the time, the output is a list of positions or velocities at the corresponding time.
 */
using FuncTime3D = std::function<std::vector<real3>(real)>;

/** Add constraints on the positions and velocities of given particles of a ParticleVector.
    The forces required to keep the particles along the given constrains are recorded and
    reported via AnchorParticlesStatsPlugin.

    \note This should not be used with RigidObjectVector.
    \note This was designed to work with ObjectVectors containing a single object, on a single rank.
          Using a plain ParticleVector might not work since particles will be reordered.
 */
class AnchorParticlesPlugin : public SimulationPlugin
{
public:
    /** Create a AnchorParticlesPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector that contains the particles of interest.
        \param [in] positions The constrains on the positions.
        \param [in] velocities The constrains on the velocities.
        \param [in] pids The concerned particle ids (starting from 0). See the restrictions in the class docs.
        \param [in] reportEvery Statistics (forces) will be sent to the AnchorParticlesStatsPlugin every this number of steps.
     */
    AnchorParticlesPlugin(const MirState *state, std::string name, std::string pvName,
                          FuncTime3D positions, FuncTime3D velocities,
                          std::vector<int> pids, int reportEvery);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    ParticleVector *pv_;

    FuncTime3D positions_;
    FuncTime3D velocities_;

    PinnedBuffer<double3> forces_;
    PinnedBuffer<real3> posBuffer_, velBuffer_;
    PinnedBuffer<int> pids_;

    int nsamples_ {0};
    int reportEvery_;
    std::vector<char> sendBuffer_;
};



/** Postprocessing side of AnchorParticlesPlugin.
    Reports the forces required to achieve the constrains in a csv file.
 */
class AnchorParticlesStatsPlugin : public PostprocessPlugin
{
public:
    /** Create a AnchorParticlesStatsPlugin object.
        \param [in] name The name of the plugin.
        \param [in] path The directory to which the stats will be dumped. Will create a single file `path/<pv_name>.csv`.
     */
    AnchorParticlesStatsPlugin(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    bool activated_;
    std::string path_;

    FileWrapper fout_;
};

} // namespace mirheo
