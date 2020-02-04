#include "plugins.h"

#include <mirheo/core/logger.h>

namespace mirheo
{

Plugin::Plugin() :
    comm(MPI_COMM_NULL),
    interComm(MPI_COMM_NULL)
{}

Plugin::~Plugin()
{
    if (comm != MPI_COMM_NULL)
        MPI_Check(MPI_Comm_free(&comm));
}

void Plugin::handshake() {}

void Plugin::setTag(int tag)
{
    tag_ = tag;
}

void Plugin::_setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    MPI_Check( MPI_Comm_dup(comm, &this->comm) );
    this->interComm = interComm;
    
    MPI_Check( MPI_Comm_rank(this->comm, &rank) );
    MPI_Check( MPI_Comm_size(this->comm, &nranks) );
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

void SimulationPlugin::beforeCellLists            (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::beforeForces               (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::beforeIntegration          (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::afterIntegration           (__UNUSED cudaStream_t stream) {}
void SimulationPlugin::beforeParticleDistribution (__UNUSED cudaStream_t stream) {}

void SimulationPlugin::serializeAndSend (__UNUSED cudaStream_t stream) {}


void SimulationPlugin::setup(__UNUSED Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    debug("Setting up simulation plugin '%s', MPI tags are (%d, %d)", getCName(), _sizeTag(), _dataTag());
    _setup(comm, interComm);
}

void SimulationPlugin::finalize()
{
    debug3("Plugin %s is finishing all the communications", getCName());
    waitPrevSend();
}

void SimulationPlugin::waitPrevSend()
{
    MPI_Check( MPI_Wait(&sizeReq_, MPI_STATUS_IGNORE) );
    MPI_Check( MPI_Wait(&dataReq_, MPI_STATUS_IGNORE) );
    sizeReq_ = MPI_REQUEST_NULL;
    dataReq_ = MPI_REQUEST_NULL;
}

void SimulationPlugin::send(const std::vector<char>& data)
{
    send(data.data(), data.size());
}

void SimulationPlugin::send(const void *data, size_t sizeInBytes)
{
    // So that async Isend of the size works on
    // valid address
    localSendSize_ = static_cast<int>(sizeInBytes);

    waitPrevSend();
        
    debug2("Plugin '%s' is sending the data (%d bytes)", getCName(), sizeInBytes);
    MPI_Check( MPI_Issend(&localSendSize_, 1, MPI_INT,  rank, _sizeTag(), interComm, &sizeReq_) );
    MPI_Check( MPI_Issend(data, static_cast<int>(sizeInBytes), MPI_BYTE, rank, _dataTag(), interComm, &dataReq_) );
}



// PostprocessPlugin

PostprocessPlugin::PostprocessPlugin(const std::string& name) :
    Plugin(),
    MirObject(name)
{}

PostprocessPlugin::~PostprocessPlugin() = default;

MPI_Request PostprocessPlugin::waitData()
{
    MPI_Request req;
    MPI_Check( MPI_Irecv(&size, 1, MPI_INT, rank, _sizeTag(), interComm, &req) );
    return req;
}

void PostprocessPlugin::recv()
{
    data.resize(size);
    MPI_Status status;
    int count;
    MPI_Check( MPI_Recv(data.data(), size, MPI_BYTE, rank, _dataTag(), interComm, &status) );
    MPI_Check( MPI_Get_count(&status, MPI_BYTE, &count) );

    if (count != size)
        error("Plugin '%s' was going to receive %d bytes, but actually got %d. That may be fatal",
              getCName(), size, count);

    debug3("Plugin '%s' has received the data (%d bytes)", getCName(), count);
}

void PostprocessPlugin::deserialize() {}

void PostprocessPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    debug("Setting up postproc plugin '%s', MPI tags are (%d, %d)", getCName(), _sizeTag(), _dataTag());
    _setup(comm, interComm);
}

} // namespace mirheo
