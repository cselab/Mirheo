#include "interface.h"

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
    this->tag = tag;
}

void Plugin::_setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    MPI_Check( MPI_Comm_dup(comm, &this->comm) );
    this->interComm = interComm;
    
    MPI_Check( MPI_Comm_rank(this->comm, &rank) );
    MPI_Check( MPI_Comm_size(this->comm, &nranks) );
}

int Plugin::_sizeTag() const {_checkTag(); return 2 * tag + 0;}
int Plugin::_dataTag() const {_checkTag(); return 2 * tag + 1;}

void Plugin::_checkTag() const
{
    if (tag == UNINITIALIZED_TAG)
        die("plugin tag is uninitialized");
}


SimulationPlugin::SimulationPlugin(const MirState *state, std::string name) :
    Plugin(),
    MirSimulationObject(state, name),
    sizeReq(MPI_REQUEST_NULL),
    dataReq(MPI_REQUEST_NULL)
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
    debug("Setting up simulation plugin '%s', MPI tags are (%d, %d)", name.c_str(), _sizeTag(), _dataTag());
    _setup(comm, interComm);
}

void SimulationPlugin::finalize()
{
    debug3("Plugin %s is finishing all the communications", name.c_str());
    waitPrevSend();
}

void SimulationPlugin::waitPrevSend()
{
    MPI_Check( MPI_Wait(&sizeReq, MPI_STATUS_IGNORE) );
    MPI_Check( MPI_Wait(&dataReq, MPI_STATUS_IGNORE) );
    sizeReq = MPI_REQUEST_NULL;
    dataReq = MPI_REQUEST_NULL;
}

void SimulationPlugin::send(const std::vector<char>& data)
{
    send(data.data(), data.size());
}

void SimulationPlugin::send(const void* data, int sizeInBytes)
{
    // So that async Isend of the size works on
    // valid address
    localSendSize = sizeInBytes;

    waitPrevSend();
        
    debug2("Plugin '%s' is sending the data (%d bytes)", name.c_str(), sizeInBytes);
    MPI_Check( MPI_Issend(&localSendSize, 1, MPI_INT,  rank, _sizeTag(), interComm, &sizeReq) );
    MPI_Check( MPI_Issend(data, sizeInBytes, MPI_BYTE, rank, _dataTag(), interComm, &dataReq) );
}



// PostprocessPlugin

PostprocessPlugin::PostprocessPlugin(std::string name) :
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
              name.c_str(), size, count);

    debug3("Plugin '%s' has received the data (%d bytes)", name.c_str(), count);
}

void PostprocessPlugin::deserialize() {}

void PostprocessPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    debug("Setting up postproc plugin '%s', MPI tags are (%d, %d)", name.c_str(), _sizeTag(), _dataTag());
    _setup(comm, interComm);
}

} // namespace mirheo
