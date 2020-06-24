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

class SimulationStats : public SimulationPlugin
{
public:
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

class PostprocessStats : public PostprocessPlugin
{
public:
    /** \brief Construct a `PostprocessStats` plugin.

        \param state MirState object
        \param name Plugin name
        \param fetchEvery Every how many time steps to print statistics?
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
