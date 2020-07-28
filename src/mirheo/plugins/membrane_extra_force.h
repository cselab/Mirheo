// Copyright 2020 ETH Zurich. All Rights Reserved.
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

/** Add external, constant forces to each vertex of a membrane.
    This was designed for a single membrane in order to model cell stretching.
 */
class MembraneExtraForcePlugin : public SimulationPlugin
{
public:
    /** Create a MembraneExtraForcePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the MembraneVector to dump.
        \param [in] forces List of forces. Must have the same size as the number of vertices.
    */
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
