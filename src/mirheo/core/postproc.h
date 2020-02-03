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
    Postprocess(MPI_Comm& comm, MPI_Comm& interComm, std::string checkpointFolder = "restart/");
    ~Postprocess();

    void registerPlugin(std::shared_ptr<PostprocessPlugin> plugin, int tag);
    void run();
    void init();

    void restart   (const std::string& folder);
    void checkpoint(int checkpointId);
    Config getConfig() const override;

private:
    MPI_Request listenSimulation(int tag, int *msg) const;
    
    using MirObject::restart;
    using MirObject::checkpoint;

private:
    MPI_Comm comm;
    MPI_Comm interComm;
    
    std::vector< std::shared_ptr<PostprocessPlugin> > plugins;

    std::string checkpointFolder;
};

} // namespace mirheo
