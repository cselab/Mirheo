#pragma once

#include <core/ymero_object.h>
#include <plugins/interface.h>

#include <memory>
#include <mpi.h>

class Postprocess : YmrObject
{
public:
    Postprocess(MPI_Comm& comm, MPI_Comm& interComm, std::string checkpointFolder = "restart/");
    ~Postprocess();

    void registerPlugin( std::shared_ptr<PostprocessPlugin> plugin );
    void run();
    void init();    

    void restart   (std::string folder);
    void checkpoint(int checkpointId);

private:
    MPI_Request listenSimulation(int tag, int *msg) const;
    
    using YmrObject::restart;
    using YmrObject::checkpoint;

private:
    MPI_Comm comm;
    MPI_Comm interComm;
    
    std::vector< std::shared_ptr<PostprocessPlugin> > plugins;

    std::string restartFolder, checkpointFolder;
};
