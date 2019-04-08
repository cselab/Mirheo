#pragma once

#include <core/logger.h>
#include <core/ymero_object.h>

#include <mpi.h>
#include <vector>

class Simulation;

class Plugin
{    
public:
    Plugin();
    virtual ~Plugin();
    
    virtual void handshake();
    virtual void talk();

protected:
    MPI_Comm comm, interComm;
    int rank, nranks;

    std::hash<std::string> nameHash;
    
    // limitation by CrayMPI (wtf Cray???)
    static const int MaxTag = 16767;

    int _tag(const std::string& name);
    void _setup(const MPI_Comm& comm, const MPI_Comm& interComm);
};

class SimulationPlugin : public Plugin, public YmrSimulationObject
{
public:
    SimulationPlugin(const YmrState *state, std::string name);
    virtual ~SimulationPlugin();

    virtual void beforeCellLists            (cudaStream_t stream);
    virtual void beforeForces               (cudaStream_t stream);
    virtual void beforeIntegration          (cudaStream_t stream);
    virtual void afterIntegration           (cudaStream_t stream);
    virtual void beforeParticleDistribution (cudaStream_t stream);

    virtual void serializeAndSend (cudaStream_t stream);

    virtual bool needPostproc() = 0;

    virtual void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm);
    virtual void finalize();    

protected:
    int localSendSize;
    MPI_Request sizeReq, dataReq;

    int _tag();
    
    void waitPrevSend();
    void send(const std::vector<char>& data);
    void send(const void* data, int sizeInBytes);
};




class PostprocessPlugin : public Plugin, public YmrObject
{
public:
    PostprocessPlugin(std::string name);
    virtual ~PostprocessPlugin();

    MPI_Request waitData();
    void recv();
    
    virtual void deserialize(MPI_Status& stat);

    virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm);    

protected:

    int _tag();
    
    std::vector<char> data;
    int size;    
};





