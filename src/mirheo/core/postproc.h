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
    void saveSnapshotAndRegister(Saver&) override;

protected:
    ConfigObject _saveSnapshot(Saver&, const std::string& typeName);

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
