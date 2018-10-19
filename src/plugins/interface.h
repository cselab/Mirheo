#pragma once

#include <mpi.h>
#include <string>
#include <core/logger.h>
#include <vector>

class Simulation;

class SimulationPlugin
{
public:
    std::string name;

    SimulationPlugin(std::string name);

    virtual void beforeForces               (cudaStream_t stream);
    virtual void beforeIntegration          (cudaStream_t stream);
    virtual void afterIntegration           (cudaStream_t stream);
    virtual void beforeParticleDistribution (cudaStream_t stream);

    virtual void serializeAndSend (cudaStream_t stream);
    virtual void handshake();
    virtual void talk();

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
    MPI_Comm comm;
    MPI_Comm interComm;
    int rank, nranks;
    int localSendSize;
    MPI_Request sizeReq, dataReq;

    float currentTime;
    int currentTimeStep;

    std::hash<std::string> nameHash;

    int tag();    
    void waitPrevSend();
    void send(const std::vector<char>& data);
    void send(const void* data, int sizeInBytes);
};




class PostprocessPlugin
{
public:
    std::string name;

    PostprocessPlugin(std::string name);

    MPI_Request waitData();
    void recv();
    
    virtual void deserialize(MPI_Status& stat);
    virtual void handshake();
    virtual void talk();

    virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm);

    virtual ~PostprocessPlugin() = default;

protected:
    MPI_Comm comm, interComm;
    int rank, nranks;
    std::vector<char> data;
    int size;

    std::hash<std::string> nameHash;

    int tag();
};





