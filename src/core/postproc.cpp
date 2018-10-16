#include "postproc.h"

#include <core/logger.h>

#include <vector>
#include <mpi.h>

Postprocess::Postprocess(MPI_Comm& comm, MPI_Comm& interComm) : comm(comm), interComm(interComm)
{
    info("Postprocessing initialized");
}

void Postprocess::registerPlugin(std::shared_ptr<PostprocessPlugin> plugin)
{
    info("New plugin registered: %s", plugin->name.c_str());
    plugins.push_back( std::move(plugin) );
}

void Postprocess::init()
{
    for (auto& pl : plugins)
    {
        debug("Setup and handshake of %s", pl->name.c_str());
        pl->setup(comm, interComm);
        pl->handshake();
    }
}

std::vector<int> findGloballyReady(std::vector<MPI_Request>& requests, std::vector<MPI_Status>& statuses, MPI_Comm comm)
{
    int index;
    MPI_Status stat;
    MPI_Check( MPI_Waitany(requests.size(), requests.data(), &index, &stat) );
    statuses[index] = stat;    

    std::vector<int> mask(requests.size(), 0);
    mask[index] = 1;
    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, mask.data(), mask.size(), MPI_INT, MPI_MAX, comm) );

    std::vector<int> ids;
    for (int i=0; i<mask.size(); ++i)
        if (mask[i] > 0)
        {
            ids.push_back(i);
            if (requests[i] != MPI_REQUEST_NULL)
                MPI_Check( MPI_Wait(&requests[i], &statuses[i]) );
        }
    
    return ids;
}

void Postprocess::run()
{
    // Stopping condition
    const int tag = 424242;

    int dummy = 0;
    int rank;

    MPI_Check( MPI_Comm_rank(comm, &rank) );

    MPI_Request endReq;
    MPI_Check( MPI_Irecv(&dummy, 1, MPI_INT, rank, tag, interComm, &endReq) );

    std::vector<MPI_Request> requests;
    for (auto& pl : plugins)
        requests.push_back(pl->waitData());
    requests.push_back(endReq);
    std::vector<MPI_Status> statuses(requests.size());
    
    info("Postprocess is listening to messages now");
    while (true)
    {
        auto readyIds = findGloballyReady(requests, statuses, comm);

        for (auto index : readyIds)
        {
            if (index == plugins.size())
            {
                if (dummy != -1)
                    die("Something went terribly wrong");
    
                info("Postprocess got a stopping message and will stop now");    
                
                for (int i=0; i<plugins.size(); i++)
                    MPI_Check( MPI_Cancel(requests.data() + i) );
                
                return;
            }
        
            debug2("Postprocess got a request from plugin '%s', executing now", plugins[index]->name.c_str());
            plugins[index]->recv();
            plugins[index]->deserialize(statuses[index]);
            requests[index] = plugins[index]->waitData();
        }
    }
}

