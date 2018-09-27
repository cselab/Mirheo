#pragma once

#include <mpi.h>
#include <string>
#include <core/logger.h>
#include <vector>

class Simulation;

// TODO: variable size messages

class SimulationPlugin
{
public:
    std::string name;

    SimulationPlugin(std::string name) :
        name(name), req(MPI_REQUEST_NULL)
    {}

    virtual void beforeForces               (cudaStream_t stream) {};
    virtual void beforeIntegration          (cudaStream_t stream) {};
    virtual void afterIntegration           (cudaStream_t stream) {};
    virtual void beforeParticleDistribution (cudaStream_t stream) {};

    virtual void serializeAndSend (cudaStream_t stream) {};
    virtual void handshake() {};
    virtual void talk() {};

    virtual bool needPostproc() = 0;

    /// Save handler state
    virtual void checkpoint(MPI_Comm& comm, std::string path) {}
    /// Restore handler state
    virtual void restart(MPI_Comm& comm, std::string path) {}


    void setTime(float t, int tstep)
    {
        currentTime = t;
        currentTimeStep = tstep;
    }

    virtual void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
    {
        this->sim = sim;

        MPI_Check( MPI_Comm_dup(comm, &this->comm) );
        this->interComm = interComm;

        MPI_Check( MPI_Comm_rank(this->comm, &rank) );
        MPI_Check( MPI_Comm_size(this->comm, &nranks) );
    }

    virtual void finalize()
    {
        debug3("Plugin %s is finishing all the communications", name.c_str());
        MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    }

    virtual ~SimulationPlugin() = default;

protected:
    Simulation* sim;
    MPI_Comm comm;
    MPI_Comm interComm;
    int rank, nranks;
    MPI_Request req;

    float currentTime;
    int currentTimeStep;

    std::hash<std::string> nameHash;
    int tag()
    {
        return (int)( nameHash(name) % 16767 );
    }
    
    void waitPrevSend()
    {
        MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
        req = MPI_REQUEST_NULL;
    }

    void send(const std::vector<char>& data)
    {
        send(data.data(), data.size());
    }

    void send(const void* data, int sizeInBytes)
    {
        waitPrevSend();
        
        debug2("Plugin '%s' has is sending the data (%d bytes)", name.c_str(), sizeInBytes);
        MPI_Check( MPI_Send(&sizeInBytes, 1, MPI_INT, rank, 2*tag(), interComm) );
        MPI_Check( MPI_Isend(data, sizeInBytes, MPI_BYTE, rank, 2*tag()+1, interComm, &req) );
    }
};

class PostprocessPlugin
{
public:
    std::string name;

    PostprocessPlugin(std::string name) : name(name) { }

    MPI_Request waitData()
    {
        MPI_Request req;
        MPI_Check( MPI_Irecv(&size, 1, MPI_INT, rank, 2*tag(), interComm, &req) );
        return req;
    }

    void recv()
    {
        data.resize(size);
        MPI_Status status;
        int count;
        MPI_Check( MPI_Recv(data.data(), size, MPI_BYTE, rank, 2*tag()+1, interComm, &status) );
        MPI_Check( MPI_Get_count(&status, MPI_BYTE, &count) );

        if (count != size)
            error("Plugin '%s' was going to receive %d bytes, but actually got %d. That may be fatal",
                    name.c_str(), size, count);

        debug3("Plugin %s has received the data (%d bytes)", name.c_str(), count);
    }

    virtual void deserialize(MPI_Status& stat) {};
    virtual void handshake() {};
    virtual void talk() {};

    virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm)
    {
        MPI_Check( MPI_Comm_dup(comm, &this->comm) );
        this->interComm = interComm;

        MPI_Check( MPI_Comm_rank(this->comm, &rank) );
        MPI_Check( MPI_Comm_size(this->comm, &nranks) );
    }

    virtual ~PostprocessPlugin() = default;


protected:
    MPI_Comm comm, interComm;
    int rank, nranks;
    std::vector<char> data;
    int size;

    std::hash<std::string> nameHash;
    int tag()
    {
        return (int)( nameHash(name) % 16767 );
    }
};





