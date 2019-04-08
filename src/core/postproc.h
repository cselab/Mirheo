#pragma once

#include <plugins/interface.h>

#include <memory>
#include <mpi.h>

class Postprocess
{
public:
    Postprocess(MPI_Comm& comm, MPI_Comm& interComm);
    void registerPlugin( std::shared_ptr<PostprocessPlugin> plugin );
    void run();
    void init();
    
    // TODO complete this
//     void restart   (std::string folder);
//     void checkpoint(std::string folder);

private:
    MPI_Request listenSimulation(int tag, int *msg) const;

private:
    MPI_Comm comm;
    MPI_Comm interComm;
    
    std::vector< std::shared_ptr<PostprocessPlugin> > plugins;
    std::vector<MPI_Request> requests;
};
