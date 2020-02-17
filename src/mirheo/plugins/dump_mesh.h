#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class ObjectVector;
class CellList;

class MeshPlugin : public SimulationPlugin
{
public:
    MeshPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery);

    /** \brief Construct a \c MeshPlugin simulation plugin object from its snapshot.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the plugin.
     */
    MeshPlugin(const MirState *state, Loader& loader, const ConfigObject& config);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    /** \brief Create a \c ConfigObject describing the plugin state and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c MeshPlugin.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /** \brief Implementation of snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    std::string ovName_;
    int dumpEvery_;

    std::vector<char> sendBuffer_;
    std::vector<real3> vertices_;
    PinnedBuffer<real4>* srcVerts_;

    ObjectVector *ov_;
};


class MeshDumper : public PostprocessPlugin
{
public:
    MeshDumper(std::string name, std::string path);

    /** \brief Construct a \c MeshDumper postprocess plugin object from its snapshot.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the plugin.
     */
    MeshDumper(Loader& loader, const ConfigObject& config);

    ~MeshDumper();

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;

    /** \brief Create a \c ConfigObject describing the plugin state and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c MeshDumper.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /** \brief Implementation of snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    std::string path_;

    bool activated_{true};

    std::vector<int3> connectivity_;
    std::vector<real3> vertices_;
};

} // namespace mirheo
