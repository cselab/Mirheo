#pragma once

#include <mpi.h>
#include <string>
#include <core/logger.h>
#include <vector>

class Simulation;

class Plugin
{
public:
    std::string name;

    Plugin(std::string name);
    
    virtual void handshake();
    virtual void talk();

    virtual ~Plugin() = default;

protected:
    MPI_Comm comm, interComm;
    int rank, nranks;

    std::hash<std::string> nameHash;

    int  _tag();
    void _setup(const MPI_Comm& comm, const MPI_Comm& interComm);
};



class SimulationPlugin : public Plugin
{
public:
    SimulationPlugin(std::string name);

    virtual void beforeForces               (cudaStream_t stream);
    virtual void beforeIntegration          (cudaStream_t stream);
    virtual void afterIntegration           (cudaStream_t stream);
    virtual void beforeParticleDistribution (cudaStream_t stream);

    virtual void serializeAndSend (cudaStream_t stream);

    virtual bool needPostproc() = 0;

    /// Save handler state
    virtual void checkpoint(MPI_Comm& comm, std::string path);
    /// Restore handler state
    virtual void restart(MPI_Comm& comm, std::string path);


    void setTime(float t, int tstep);
    virtual void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm);
    virtual void finalize();

    virtual ~SimulationPlugin() = default;

protected:
    Simulation* sim;
    int localSendSize;
    MPI_Request sizeReq, dataReq;

    float currentTime;
    int currentTimeStep;

    void waitPrevSend();
    void send(const std::vector<char>& data);
    void send(const void* data, int sizeInBytes);
};




class PostprocessPlugin : public Plugin
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





