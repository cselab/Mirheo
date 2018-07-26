#pragma once

#include <mpi.h>
#include <string>
#include <core/logger.h>
#include <vector>

class Simulation;

// TODO: variable size messages

enum {
    PLUGINS_INTERCOMM_TAG_DATA,
    PLUGINS_INTERCOMM_TAG_SIZE,
};

class SimulationPlugin
{
public:
    std::string name;

    SimulationPlugin(std::string name) :
        name(name), req(MPI_REQUEST_NULL)
    {}

    virtual void beforeForces     (cudaStream_t stream) {};
    virtual void beforeIntegration(cudaStream_t stream) {};
    virtual void afterIntegration (cudaStream_t stream) {};

    virtual void serializeAndSend (cudaStream_t stream) {};
    virtual void handshake() {};
    virtual void talk() {};

    virtual bool needPostproc() = 0;

    void setTime(float t, int tstep)
    {
        currentTime = t;
        currentTimeStep = tstep;
    }

    virtual void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
    {
        this->sim = sim;

        MPI_Check( MPI_Comm_dup(comm,      &this->comm     ) );
        MPI_Check( MPI_Comm_dup(interComm, &this->interComm) );

        MPI_Check( MPI_Comm_rank(this->comm, &rank) );
        MPI_Check( MPI_Comm_size(this->comm, &nranks) );
    }

    virtual void finalize()
    {
        debug3("Plugin %s is finishing all the communications", name.c_str());
        MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
        MPI_Check( MPI_Comm_free(&comm     ) );
        MPI_Check( MPI_Comm_free(&interComm) );
    }

    virtual ~SimulationPlugin() = default;

protected:
    Simulation* sim;
    MPI_Comm comm, interComm;
    int rank, nranks;
    MPI_Request req;

    float currentTime;
    int currentTimeStep;

    void send(const std::vector<char>& data)
    {
        send(data.data(), data.size());
    }

    void send(const void* data, int sizeInBytes)
    {
        debug3("Plugin %s is sending now", name.c_str());
        MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );

        MPI_Check( MPI_Ssend(&sizeInBytes, 1,    MPI_INT,  rank, PLUGINS_INTERCOMM_TAG_SIZE, interComm) );
        MPI_Check( MPI_Issend(data, sizeInBytes, MPI_BYTE, rank, PLUGINS_INTERCOMM_TAG_DATA, interComm, &req) );

        debug3("Plugin %s has sent the data (%d bytes)", name.c_str(), sizeInBytes);
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
        MPI_Check( MPI_Irecv(&size, 1, MPI_INT, rank, PLUGINS_INTERCOMM_TAG_SIZE, interComm, &req) );
        return req;
    }

    void recv()
    {
        data.resize(size);
        MPI_Status status;
        int count;
        MPI_Check( MPI_Recv(data.data(), size, MPI_BYTE, rank, PLUGINS_INTERCOMM_TAG_DATA, interComm, &status) );
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
        MPI_Check( MPI_Comm_dup(comm,      &this->comm     ) );
        MPI_Check( MPI_Comm_dup(interComm, &this->interComm) );

        MPI_Check( MPI_Comm_rank(this->comm, &rank) );
        MPI_Check( MPI_Comm_size(this->comm, &nranks) );
    }

    virtual ~PostprocessPlugin() = default;


protected:
    MPI_Comm comm, interComm;
    int rank, nranks;
    std::vector<char> data;
    int size;
};





