#include "postproc.h"

#include <core/logger.h>
#include <core/utils/common.h>

#include <mpi.h>
#include <vector>

Postprocess::Postprocess(MPI_Comm& comm, MPI_Comm& interComm, std::string checkpointFolder) :
    MirObject("postprocess"),
    comm(comm),
    interComm(interComm),
    checkpointFolder(checkpointFolder)
{
    info("Postprocessing initialized");
}

Postprocess::~Postprocess() = default;

void Postprocess::registerPlugin(std::shared_ptr<PostprocessPlugin> plugin, int tag)
{
    info("New plugin registered: %s", plugin->name.c_str());
    plugin->setTag(tag);
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

static std::vector<int> findGloballyReady(std::vector<MPI_Request>& requests, std::vector<MPI_Status>& statuses, MPI_Comm comm)
{
    int index;
    MPI_Status stat;
    MPI_Check( MPI_Waitany(requests.size(), requests.data(), &index, &stat) );
    statuses[index] = stat;    

    std::vector<int> mask(requests.size(), 0);
    mask[index] = 1;
    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, mask.data(), mask.size(), MPI_INT, MPI_MAX, comm) );

    std::vector<int> ids;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i] > 0)
        {
            ids.push_back(i);
            if (requests[i] != MPI_REQUEST_NULL)
                MPI_Check( MPI_Wait(&requests[i], &statuses[i]) );
        }
    
    return ids;
}

static void safeCancelAndFreeRequest(MPI_Request& req)
{
    if (req != MPI_REQUEST_NULL)
    {
        MPI_Check( MPI_Cancel(&req) );
        MPI_Check( MPI_Request_free(&req) );
    }
}

void Postprocess::run()
{
    int endMsg {0}, checkpointId {0};

    std::vector<MPI_Request> requests;
    for (auto& pl : plugins)
        requests.push_back(pl->waitData());

    const int stoppingReqIndex = requests.size();
    requests.push_back( listenSimulation(stoppingTag, &endMsg) );

    const int checkpointReqIndex = requests.size();
    requests.push_back( listenSimulation(checkpointTag, &checkpointId) );

    std::vector<MPI_Status> statuses(requests.size());
    
    info("Postprocess is listening to messages now");
    while (true)
    {
        auto readyIds = findGloballyReady(requests, statuses, comm);

        for (auto index : readyIds)
        {
            if (index == stoppingReqIndex)
            {
                if (endMsg != stoppingMsg) die("Received wrong stopping message");
    
                info("Postprocess got a stopping message and will stop now");    
                
                for (auto& req : requests)
                    safeCancelAndFreeRequest(req);
                
                return;
            }
            else if (index == checkpointReqIndex)
            {
                debug2("Postprocess got a request for checkpoint, executing now");
                checkpoint(checkpointId);
                requests[index] = listenSimulation(checkpointTag, &checkpointId);
            }
            else
            {
                debug2("Postprocess got a request from plugin '%s', executing now", plugins[index]->name.c_str());
                plugins[index]->recv();
                plugins[index]->deserialize(statuses[index]);
                requests[index] = plugins[index]->waitData();
            }
        }
    }
}

MPI_Request Postprocess::listenSimulation(int tag, int *msg) const
{
    int rank;
    MPI_Request req;
    
    MPI_Check( MPI_Comm_rank(comm, &rank) );    
    MPI_Check( MPI_Irecv(msg, 1, MPI_INT, rank, tag, interComm, &req) );

    return req;
}

void Postprocess::restart(std::string folder)
{
    restartFolder = folder;

    info("Reading postprocess state, from folder %s", restartFolder.c_str());
    
    for (auto& pl : plugins)
        pl->restart(comm, folder);    
}

void Postprocess::checkpoint(int checkpointId)
{
    info("Writing postprocess state, into folder %s", checkpointFolder.c_str());
    
    for (auto& pl : plugins)
        pl->checkpoint(comm, checkpointFolder, checkpointId);
}
