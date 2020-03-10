#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <mirheo/core/xdmf/xdmf.h>

#include <vector>
#include <string>

namespace mirheo
{

class ParticleVector;
class CellList;

class ParticleSenderPlugin : public SimulationPlugin
{
public:
    ParticleSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                         const std::vector<std::string>& channelNames);

    /// Construct a simulation plugin object from its snapshot.
    ParticleSenderPlugin(const MirState *state, Loader& loader, const ConfigObject& config);

    ~ParticleSenderPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

    std::string pvName_;
    ParticleVector *pv_;

    std::vector<char> sendBuffer_;

private:
    int dumpEvery_;

    HostBuffer<real4> positions_, velocities_;
    std::vector<std::string> channelNames_;
    DeviceBuffer<char> workSpace_;
    std::vector<HostBuffer<char>> channelData_;
};


class ParticleDumperPlugin : public PostprocessPlugin
{
public:
    ParticleDumperPlugin(std::string name, std::string path);

    /// Construct a postprocess plugin object from its snapshot.
    ParticleDumperPlugin(Loader& loader, const ConfigObject& config);

    ~ParticleDumperPlugin();

    void deserialize() override;
    void handshake() override;

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

    void _recvAndUnpack(MirState::TimeType &time, MirState::StepType& timeStamp);

protected:
    static constexpr int zeroPadding_ = 5;
    std::string path_;

    std::vector<real4> pos4_, vel4_;
    std::vector<real3> velocities_;
    std::vector<int64_t> ids_;
    std::shared_ptr<std::vector<real3>> positions_;

    std::vector<XDMF::Channel> channels_;
    std::vector<std::vector<char>> channelData_;
};

} // namespace mirheo
