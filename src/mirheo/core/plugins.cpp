// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "plugins.h"

#include <mirheo/core/logger.h>

#include <cassert>

namespace mirheo
{

Plugin::Plugin() :
    interComm_(MPI_COMM_NULL),
    rank_{-1},
    nranks_{-1}
{}

Plugin::~Plugin() = default;

void Plugin::handshake() {}

void Plugin::setTag(int tag)
{
    tag_ = tag;
}

void Plugin::_setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    if (comm_ != MPI_COMM_NULL) {
        assert(interComm_ == interComm);
        return;
    }

    MPI_Check( MPI_Comm_dup(comm, comm_.reset_and_get_address()) );
    interComm_ = interComm;

    MPI_Check( MPI_Comm_rank(comm_, &rank_) );
    MPI_Check( MPI_Comm_size(comm_, &nranks_) );
}

int Plugin::_sizeTag() const {_checkTag(); return 2 * tag_ + 0;}
int Plugin::_dataTag() const {_checkTag(); return 2 * tag_ + 1;}

void Plugin::_checkTag() const
{
    if (tag_ == invalidTag)
        die("plugin tag is uninitialized");
}


SimulationPlugin::SimulationPlugin(const MirState *state, const std::string& name) :
    Plugin(),
    MirSimulationObject(state, name),
    sizeReq_(MPI_REQUEST_NULL),
    dataReq_(MPI_REQUEST_NULL)
{}

SimulationPlugin::~SimulationPlugin() = default;

void SimulationPlugin::setup(__UNUSED Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    debug("Setting up simulation plugin '%s', MPI tags are (%d, %d)", getCName(), _sizeTag(), _dataTag());
    _setup(comm, interComm);
}

void SimulationPlugin::beforeCellLists            (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::beforeForces               (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::beforeIntegration          (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::afterIntegration           (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::beforeParticleDistribution (__UNUSED cudaStream_t stream) {}

void SimulationPlugin::serializeAndSend (__UNUSED cudaStream_t stream) {}


void SimulationPlugin::finalize()
{
    debug3("Plugin %s is finishing all the communications", getCName());
    _waitPrevSend();
}

void SimulationPlugin::_waitPrevSend()
{
    MPI_Check( MPI_Wait(&sizeReq_, MPI_STATUS_IGNORE) );
    MPI_Check( MPI_Wait(&dataReq_, MPI_STATUS_IGNORE) );
    sizeReq_ = MPI_REQUEST_NULL;
    dataReq_ = MPI_REQUEST_NULL;
}

void SimulationPlugin::_send(const std::vector<char>& data)
{
    _send(data.data(), data.size());
}

void SimulationPlugin::_send(const void *data, size_t sizeInBytes)
{
    // So that async Isend of the size works on
    // valid address
    localSendSize_ = static_cast<int>(sizeInBytes);

    _waitPrevSend();

    debug2("Plugin '%s' is sending the data (%zu bytes)", getCName(), sizeInBytes);
    MPI_Check( MPI_Issend(&localSendSize_, 1, MPI_INT,  rank_, _sizeTag(), interComm_, &sizeReq_) );
    MPI_Check( MPI_Issend(data, static_cast<int>(sizeInBytes), MPI_BYTE, rank_, _dataTag(), interComm_, &dataReq_) );
}

// PostprocessPlugin

PostprocessPlugin::PostprocessPlugin(const std::string& name) :
    Plugin(),
    MirObject(name)
{}

PostprocessPlugin::~PostprocessPlugin() = default;

void PostprocessPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    debug("Setting up postproc plugin '%s', MPI tags are (%d, %d)", getCName(), _sizeTag(), _dataTag());
    _setup(comm, interComm);
}

void PostprocessPlugin::recv()
{
    data_.resize(size_);
    MPI_Status status;
    int count;
    MPI_Check( MPI_Recv(data_.data(), size_, MPI_BYTE, rank_, _dataTag(), interComm_, &status) );
    MPI_Check( MPI_Get_count(&status, MPI_BYTE, &count) );

    if (count != size_)
        error("Plugin '%s' was going to receive %d bytes, but actually got %d. That may be fatal",
              getCName(), size_, count);

    debug3("Plugin '%s' has received the data (%d bytes)", getCName(), count);
}

MPI_Request PostprocessPlugin::waitData()
{
    MPI_Request req;
    MPI_Check( MPI_Irecv(&size_, 1, MPI_INT, rank_, _sizeTag(), interComm_, &req) );
    return req;
}

void PostprocessPlugin::deserialize() {}

} // namespace mirheo
