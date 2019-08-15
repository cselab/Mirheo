#pragma once

#include <core/mirheo_object.h>
#include <plugins/interface.h>

#include <memory>
#include <mpi.h>

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
