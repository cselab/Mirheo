#pragma once

#include <mirheo/core/mirheo_object.h>
#include <mirheo/core/plugins.h>

#include <memory>
#include <mpi.h>

namespace mirheo
{

class Postprocess : MirObject
{
public:
    Postprocess(MPI_Comm& comm, MPI_Comm& interComm, const std::string& checkpointFolder);
    ~Postprocess();

    void registerPlugin(std::shared_ptr<PostprocessPlugin> plugin, int tag);
    void run();
    void init();

    void restart   (const std::string& folder);
    void checkpoint(int checkpointId);

    /** \brief Dump all postprocess data, create a ConfigObject describing the postprocess state and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly Postprocess.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    MPI_Request listenSimulation(int tag, int *msg) const;
    
    using MirObject::restart;
    using MirObject::checkpoint;

private:
    friend Saver;

    MPI_Comm comm_;
    MPI_Comm interComm_;
    
    std::vector< std::shared_ptr<PostprocessPlugin> > plugins_;

    std::string checkpointFolder_;
};

} // namespace mirheo
