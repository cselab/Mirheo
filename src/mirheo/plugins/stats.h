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

namespace stats_plugin
{
using ReductionType = double;
using CountType = unsigned long long;
}

/** Collect global statistics of the simulation and send it to the postprocess ranks.
    Compute total linear momentum and estimate of temperature.
    Furthermore, measures average wall time of time steps.
 */
class SimulationStats : public SimulationPlugin
{
public:
    /** Create a SimulationPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] fetchEvery Send the statistics every this number of steps.
    */
    SimulationStats(const MirState *state, std::string name, int fetchEvery);

    /// Construct a simulation plugin object from its snapshot.
    SimulationStats(const MirState *state, Loader& loader, const ConfigObject& config);

    ~SimulationStats();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    int fetchEvery_;
    bool needToDump_{false};

    stats_plugin::CountType nparticles_;
    PinnedBuffer<stats_plugin::ReductionType> momentum_{3}, energy_{1};
    PinnedBuffer<real> maxvel_{1};
    std::vector<char> sendBuffer_;

    std::vector<ParticleVector*> pvs_;

    mTimer timer_;
};


/** Dump the stats sent by SimulationStats to a csv file and to the console output.
 */
class PostprocessStats : public PostprocessPlugin
{
public:
    /** Construct a PostprocessStats plugin.
        \param [in] name The name of the plugin.
        \param [in] filename The csv file name that will be dumped.
      */
    PostprocessStats(std::string name, std::string filename = std::string());

    /// Construct a postprocess plugin object from its snapshot.
    PostprocessStats(Loader& loader, const ConfigObject& config);

    void deserialize() override;

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    FileWrapper fdump_;
    std::string filename_;
};

} // namespace mirheo
