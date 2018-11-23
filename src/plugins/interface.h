#pragma once

#include <mpi.h>
#include <core/logger.h>
#include <vector>

#include "core/ymero_object.h"

class Simulation;

template<class Base>
class Plugin : public Base
{    
public:
    Plugin(std::string name) : Base(name) {};
    
    using Base::name;
    
    virtual void handshake() {};
    virtual void talk() {};

    virtual ~Plugin() = default;

protected:
    MPI_Comm comm, interComm;
    int rank, nranks;

    std::hash<std::string> nameHash;
    
    // limitation by CrayMPI (wtf Cray???)
    static const int MaxTag = 16767;

    int  _tag() { return (int)( nameHash(name) % MaxTag ); }
    void _setup(const MPI_Comm& comm, const MPI_Comm& interComm)
    {
        MPI_Check( MPI_Comm_dup(comm, &this->comm) );
        this->interComm = interComm;
        
        MPI_Check( MPI_Comm_rank(this->comm, &rank) );
        MPI_Check( MPI_Comm_size(this->comm, &nranks) );
    }
};

class SimulationPlugin : public Plugin<YmrSimulationObject>
{
public:
    SimulationPlugin(std::string name);

    virtual void beforeForces               (cudaStream_t stream);
    virtual void beforeIntegration          (cudaStream_t stream);
    virtual void afterIntegration           (cudaStream_t stream);
    virtual void beforeParticleDistribution (cudaStream_t stream);

    virtual void serializeAndSend (cudaStream_t stream);

    virtual bool needPostproc() = 0;

    void setTime(float t, int tstep);
    virtual void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm);
    virtual void finalize();

    virtual ~SimulationPlugin() = default;

protected:
    int localSendSize;
    MPI_Request sizeReq, dataReq;

    float currentTime;
    int currentTimeStep;

    void waitPrevSend();
    void send(const std::vector<char>& data);
    void send(const void* data, int sizeInBytes);
};




class PostprocessPlugin : public Plugin<YmrObject>
{
public:
    PostprocessPlugin(std::string name);

    MPI_Request waitData();
    void recv();
    
    virtual void deserialize(MPI_Status& stat);

    virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm);

    virtual ~PostprocessPlugin() = default;

protected:
    std::vector<char> data;
    int size;    
};





