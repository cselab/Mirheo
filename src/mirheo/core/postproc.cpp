// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "postproc.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/compile_options.h>
#include <mirheo/core/utils/path.h>

#include <mpi.h>

#include <cassert>
#include <vector>

namespace mirheo
{

Postprocess::Postprocess(MPI_Comm& comm, MPI_Comm& interComm,
                         const CheckpointInfo& checkpointInfo) :
    MirObject("postprocess"),
    comm_(comm),
    interComm_(interComm),
    checkpointFolder_(checkpointInfo.folder)
{
    info("Postprocessing initialized");
}

Postprocess::~Postprocess() = default;

void Postprocess::registerPlugin(std::shared_ptr<PostprocessPlugin> plugin, int tag)
{
    info("New plugin registered: %s", plugin->getCName());
    plugin->setTag(tag);
    plugins_.push_back( std::move(plugin) );
}

void Postprocess::deregisterPlugin(PostprocessPlugin *plugin)
{
    for (auto it = plugins_.begin(); it != plugins_.end(); ++it) {
        if (it->get() == plugin) {
            info("Plugin deregistered: %s", plugin->getCName());
            plugins_.erase(it);
            return;
        }
    }

    die("PostprocessPlugin %s not found", plugin->getCName());
}

void Postprocess::init()
{
    for (auto& pl : plugins_)
    {
        debug("Setup and handshake of %s", pl->getCName());
        // Both setup and handshake will be invoked each time run() is invoked.
        // The plugin should support multiple runs if possible. Otherwise, the
        // plugin should throw an exception when the run is invoked for the
        // second time.
        pl->setup(comm_, interComm_);
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
    for (auto& pl : plugins_)
        requests.push_back(pl->waitData());

    const int stoppingReqIndex = static_cast<int>(requests.size());
    requests.push_back( _listenSimulation(stoppingTag, &endMsg) );

    const int checkpointReqIndex = static_cast<int>(requests.size());
    requests.push_back( _listenSimulation(checkpointTag, &checkpointId) );

    std::vector<MPI_Status> statuses(requests.size());

    info("Postprocess is listening to messages now");
    while (true)
    {
        const auto readyIds = findGloballyReady(requests, statuses, comm_);

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
                requests[index] = _listenSimulation(checkpointTag, &checkpointId);
            }
            else
            {
                debug2("Postprocess got a request from plugin '%s', executing now", plugins_[index]->getCName());
                plugins_[index]->recv();
                plugins_[index]->deserialize();
                requests[index] = plugins_[index]->waitData();
            }
        }
    }
}

MPI_Request Postprocess::_listenSimulation(int tag, int *msg) const
{
    int rank;
    MPI_Request req;

    MPI_Check( MPI_Comm_rank(comm_, &rank) );
    MPI_Check( MPI_Irecv(msg, 1, MPI_INT, rank, tag, interComm_, &req) );

    return req;
}

void Postprocess::restart(const std::string& folder)
{
    info("Reading postprocess state, from folder %s", folder.c_str());

    for (auto& pl : plugins_)
        pl->restart(comm_, folder);
}

void Postprocess::checkpoint(int checkpointId)
{
    info("Writing postprocess state, into folder %s", checkpointFolder_.c_str());

    for (auto& pl : plugins_)
        pl->checkpoint(comm_, checkpointFolder_, checkpointId);
}


} // namespace mirheo
