// Copyright 2020 ETH Zurich. All Rights Reserved.
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

/** Send particle data to ParticleDumperPlugin.
*/
class ParticleSenderPlugin : public SimulationPlugin
{
public:
    /** Create a ParticleSenderPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to dump.
        \param [in] dumpEvery Send the data to the postprocess side every this number of steps.
        \param [in] channelNames The list of channels to send, additionally to the default positions, velocities and global ids.
     */
    ParticleSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                         const std::vector<std::string>& channelNames);

    /** Load a snapshot of the plugin.
        \param [in] state The global state of the simulation.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    ParticleSenderPlugin(const MirState *state, Loader& loader, const ConfigObject& config);

    ~ParticleSenderPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

    std::string pvName_; ///< name of the ParticleVector to dump.
    ParticleVector *pv_; ///< pointer to the ParticleVector to dump.

    std::vector<char> sendBuffer_; ///< Buffer used to send the data to the postprocess side.

private:
    int dumpEvery_;

    HostBuffer<real4> positions_, velocities_;
    std::vector<std::string> channelNames_;
    DeviceBuffer<char> workSpace_;
    std::vector<HostBuffer<char>> channelData_;
};


/** Postprocess side of ParticleSenderPlugin.
    Dump particles data to xmf + hdf5 format.
*/
class ParticleDumperPlugin : public PostprocessPlugin
{
public:
    /** Create a ParticleDumperPlugin object.
        \param [in] name The name of the plugin.
        \param [in] path Particle data will be dumped to `pathXXXXX.[xmf,h5]`.
    */
    ParticleDumperPlugin(std::string name, std::string path);

    /** Load a snapshot of the plugin.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    ParticleDumperPlugin(Loader& loader, const ConfigObject& config);

    ~ParticleDumperPlugin();

    void deserialize() override;
    void handshake() override;

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

    /** Receive and unpack the data from The simulation side.
        \param [out] time The current time.
        \param [out] timeStamp The dump id.
     */
    void _recvAndUnpack(MirState::TimeType &time, MirState::StepType& timeStamp);

protected:
    static constexpr int zeroPadding_ = 5; ///< number of zero padding for the file names.
    std::string path_; ///< base dump path.

    std::vector<real4> pos4_; ///< Received positions and half the ids.
    std::vector<real4> vel4_; ///< Received velocities and half the ids.
    std::vector<real3> velocities_; ///< Processed velocities.
    std::vector<int64_t> ids_; ///< Processed ids.
    std::shared_ptr<std::vector<real3>> positions_; ///< Processed positions.

    std::vector<XDMF::Channel> channels_; ///< List of received channel descriptions.
    std::vector<std::vector<char>> channelData_; ///< List of received channel data.
};

} // namespace mirheo
