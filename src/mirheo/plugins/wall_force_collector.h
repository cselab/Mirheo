// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/timer.h>

namespace mirheo
{

class ParticleVector;
class SDFBasedWall;

/** Compute the force exerted by particles on the walls.
    It has two contributions:
    - Interactio forces with frozen particles
    - bounce-back.
 */
class WallForceCollectorPlugin : public SimulationPlugin
{
public:
    /** Create a WallForceCollectorPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] wallName The name of the \c Wall to collect the forces from.
        \param [in] frozenPvName The name of the frozen ParticleVector assigned to the wall.
        \param [in] sampleEvery Compute forces every this number of steps, and average it in time.
        \param [in] dumpEvery Send the average forces to the postprocessing side every this number of steps.
     */
    WallForceCollectorPlugin(const MirState *state, std::string name,
                             std::string wallName, std::string frozenPvName,
                             int sampleEvery, int dumpEvery);
    ~WallForceCollectorPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    int sampleEvery_, dumpEvery_;
    int nsamples_ {0};

    std::string wallName_;
    std::string frozenPvName_;

    bool needToDump_ {false};

    SDFBasedWall *wall_;
    ParticleVector *pv_;

    PinnedBuffer<double3> *bounceForceBuffer_ {nullptr};
    PinnedBuffer<double3> pvForceBuffer_ {1};
    double3 bounceForce_ {0.0, 0.0, 0.0};
    double3 pvForce_ {0.0, 0.0, 0.0};

    std::vector<char> sendBuffer_;
};


/** Postprocess side of WallForceCollectorPlugin.
    Dump the forces to a txt file.
*/
class WallForceDumperPlugin : public PostprocessPlugin
{
public:
    /** Create a WallForceDumperPlugin.
        \param [in] name The name of the plugin.
        \param [in] filename The file to dump the stats to.
    */
    WallForceDumperPlugin(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump_;
};

} // namespace mirheo
