#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <array>
#include <string>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

class MembraneVector;

class MembraneExtraForcePlugin : public SimulationPlugin
{
public:
    MembraneExtraForcePlugin(const MirState *state, std::string name, std::string pvName, const std::vector<real3>& forces);

    /** \brief Construct a simulation plugin object from its snapshot.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the plugin.
     */
    MembraneExtraForcePlugin(const MirState *state, Loader& loader, const ConfigObject& config);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    /** \brief Create a \c ConfigObject describing the plugin state and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c MembraneExtraForcePlugin.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /** \brief Implementation of snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    std::string pvName_;
    MembraneVector *pv_;
    DeviceBuffer<Force> forces_;
};

} // namespace mirheo
