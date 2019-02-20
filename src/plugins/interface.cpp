#include "interface.h"

Plugin::Plugin() = default;
Plugin::~Plugin() = default;
    
void Plugin::handshake() {}
void Plugin::talk() {}  

int Plugin::_tag(const std::string& name)
{
    return (int)( nameHash(name) % MaxTag );
}

void Plugin::_setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    MPI_Check( MPI_Comm_dup(comm, &this->comm) );
    this->interComm = interComm;
    
    MPI_Check( MPI_Comm_rank(this->comm, &rank) );
    MPI_Check( MPI_Comm_size(this->comm, &nranks) );
}



SimulationPlugin::SimulationPlugin(const YmrState *state, std::string name) :
    Plugin(),
    YmrSimulationObject(state, name),
    sizeReq(MPI_REQUEST_NULL),
    dataReq(MPI_REQUEST_NULL)
{}

SimulationPlugin::~SimulationPlugin() = default;

void SimulationPlugin::beforeCellLists            (cudaStream_t stream) {}
void SimulationPlugin::beforeForces               (cudaStream_t stream) {}
void SimulationPlugin::beforeIntegration          (cudaStream_t stream) {}
void SimulationPlugin::afterIntegration           (cudaStream_t stream) {}
void SimulationPlugin::beforeParticleDistribution (cudaStream_t stream) {}

void SimulationPlugin::serializeAndSend (cudaStream_t stream) {}


void SimulationPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    debug("Setting up simulation plugin '%s', MPI tag is %d", name.c_str(), _tag());
    _setup(comm, interComm);
}

void SimulationPlugin::finalize()
{
    debug3("Plugin %s is finishing all the communications", name.c_str());
    MPI_Check( MPI_Wait(&sizeReq, MPI_STATUS_IGNORE) );
    MPI_Check( MPI_Wait(&dataReq, MPI_STATUS_IGNORE) );
}

int SimulationPlugin::_tag()
{
    return Plugin::_tag(name);
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
    MPI_Check( MPI_Issend(&localSendSize, 1, MPI_INT, rank, 2*_tag(), interComm, &sizeReq) );
    MPI_Check( MPI_Issend(data, sizeInBytes, MPI_BYTE, rank, 2*_tag()+1, interComm, &dataReq) );
}



// PostprocessPlugin

PostprocessPlugin::PostprocessPlugin(std::string name) :
    Plugin(), YmrObject(name)
{}

PostprocessPlugin::~PostprocessPlugin() = default;

MPI_Request PostprocessPlugin::waitData()
{
    MPI_Request req;
    MPI_Check( MPI_Irecv(&size, 1, MPI_INT, rank, 2*_tag(), interComm, &req) );
    return req;
}

void PostprocessPlugin::recv()
{
    data.resize(size);
    MPI_Status status;
    int count;
    MPI_Check( MPI_Recv(data.data(), size, MPI_BYTE, rank, 2*_tag()+1, interComm, &status) );
    MPI_Check( MPI_Get_count(&status, MPI_BYTE, &count) );

    if (count != size)
        error("Plugin '%s' was going to receive %d bytes, but actually got %d. That may be fatal",
              name.c_str(), size, count);

    debug3("Plugin '%s' has received the data (%d bytes)", name.c_str(), count);
}

void PostprocessPlugin::deserialize(MPI_Status& stat) {};

void PostprocessPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    debug("Setting up postproc plugin '%s', MPI tag is %d", name.c_str(), _tag());
    _setup(comm, interComm);
}

int PostprocessPlugin::_tag()
{
    return Plugin::_tag(name);
}




