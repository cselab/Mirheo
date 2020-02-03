#include "postproc.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/config.h>

#include <mpi.h>
#include <vector>

namespace mirheo
{

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
    MPI_Check( MPI_Waitany((int) requests.size(), requests.data(), &index, &stat) );
    statuses[index] = stat;    

    std::vector<int> mask(requests.size(), 0);
    mask[index] = 1;
    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, mask.data(), (int) mask.size(), MPI_INT, MPI_MAX, comm) );

    std::vector<int> ids;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i] > 0)
        {
            ids.push_back(static_cast<int>(i));
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

    const int stoppingReqIndex = static_cast<int>(requests.size());
    requests.push_back( listenSimulation(stoppingTag, &endMsg) );

    const int checkpointReqIndex = static_cast<int>(requests.size());
    requests.push_back( listenSimulation(checkpointTag, &checkpointId) );

    std::vector<MPI_Status> statuses(requests.size());
    
    info("Postprocess is listening to messages now");
    while (true)
    {
        const auto readyIds = findGloballyReady(requests, statuses, comm);

        for (const auto& index : readyIds)
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
                plugins[index]->deserialize();
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

void Postprocess::restart(const std::string& folder)
{
    info("Reading postprocess state, from folder %s", folder.c_str());
    
    for (auto& pl : plugins)
        pl->restart(comm, folder);    
}

void Postprocess::checkpoint(int checkpointId)
{
    info("Writing postprocess state, into folder %s", checkpointFolder.c_str());
    
    for (auto& pl : plugins)
        pl->checkpoint(comm, checkpointFolder, checkpointId);
}

Config Postprocess::getConfig() const {
    Config::List pluginsConfig;
    pluginsConfig.reserve(plugins.size());
    for (const auto &plugin : plugins)
        pluginsConfig.push_back(plugin->getConfig());
    return Config::Dictionary{
        {"name", name},
        {"checkpointFolder", checkpointFolder},
        {"plugins", std::move(pluginsConfig)},
    };
}

} // namespace mirheo
