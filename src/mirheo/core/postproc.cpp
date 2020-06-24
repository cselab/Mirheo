// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "postproc.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/compile_options.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/folders.h>

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
    checkpointFolder_(checkpointInfo.folder),
    checkpointMechanism_(checkpointInfo.mechanism)
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

void Postprocess::init()
{
    for (auto& pl : plugins_)
    {
        debug("Setup and handshake of %s", pl->getCName());
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
                if (checkpointMechanism_ == CheckpointMechanism::Checkpoint)
                    checkpoint(checkpointId);
                else
                    snapshot(createSnapshotPath(checkpointFolder_, checkpointId));
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

/// Prepare a ConfigObject listing all compilation options that affect the output format.
static ConfigObject compileOptionsToConfig(Saver& saver) {
    // If updating this, don't forget to update snapshot.cpp:checkCompilationOptions.
    return ConfigObject{
        {"useDouble", saver(compile_options.useDouble)},
    };
}

void Postprocess::snapshot(const std::string& path)
{
    // Prepare context and the saver.
    SaverContext context;
    context.path = path;
    context.groupComm = comm_;
    Saver saver{&context};

    // Folder creation barrier.
    MPI_Barrier(interComm_);

    // All ranks participate in storing the Postprocess object and its plugins.
    saver.registerObject<Postprocess>(this, _saveSnapshot(saver, "Postprocess"));

    int rank;
    MPI_Comm_rank(comm_, &rank);
    if (rank == 0) {
        // Get the JSON object from the simulation side.
        int size;
        MPI_Check( MPI_Recv(&size, 1, MPI_INT, 0, snapshotTag, interComm_, MPI_STATUS_IGNORE) );
        std::string simJson(size, '_');
        MPI_Check( MPI_Recv(const_cast<char *>(simJson.data()), size, MPI_CHAR,
                            0, snapshotTag, interComm_, MPI_STATUS_IGNORE) );

        // Postprocessing side will be merged into the simulation side.
        ConfigObject all = configFromJSON(simJson).getObject();
        ConfigObject &post = saver.getConfig();
        assert(post.size() == 1 || post.size() == 2);  // Only plugins (optionally) and Postprocess.
        auto it = post.find("PostprocessPlugin");
        if (it != post.end()) {
            // Insert before `SimulationPlugin` for cosmetic reasons.
            all.unsafe_insert(all.find("SimulationPlugin"),
                              "PostprocessPlugin", std::move(it->second));
        }
        all.unsafe_insert(all.find("Simulation"),
                          "Postprocess", std::move(post["Postprocess"]));

        // Store compile options as a special category.
        all.unsafe_insert("CompileOptions", compileOptionsToConfig(saver));

        // Save the config.json.
        std::string content = ConfigValue(std::move(all)).toJSONString() + '\n';
        FileWrapper f(joinPaths(path, "config.json"), "w");
        fwrite(content.data(), 1, content.size(), f.get());
    }
}

ConfigObject Postprocess::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = MirObject::_saveSnapshot(saver, "Postprocess", typeName);
    config.emplace("checkpointFolder", saver(checkpointFolder_));
    config.emplace("plugins",          saver(plugins_));
    return config;
}

} // namespace mirheo
