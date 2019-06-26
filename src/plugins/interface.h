#pragma once

#include <core/logger.h>
#include <core/mirheo_object.h>

#include <mpi.h>
#include <vector>

class Simulation;

class Plugin
{    
public:
    Plugin();
    virtual ~Plugin();
    
    virtual void handshake();

    void setTag(int tag);
    
protected:
    MPI_Comm comm, interComm;
    int rank, nranks;
    
    void _setup(const MPI_Comm& comm, const MPI_Comm& interComm);

    int _sizeTag() const;
    int _dataTag() const;

private:
    void _checkTag() const;
    enum {UNINITIALIZED_TAG = -1};
    int tag {UNINITIALIZED_TAG};
};

class SimulationPlugin : public Plugin, public MirSimulationObject
{
public:
    SimulationPlugin(const MirState *state, std::string name);
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

    void waitPrevSend();
    void send(const std::vector<char>& data);
    void send(const void *data, int sizeInBytes);
};




class PostprocessPlugin : public Plugin, public MirObject
{
public:
    PostprocessPlugin(std::string name);
    virtual ~PostprocessPlugin();

    MPI_Request waitData();
    void recv();
    
    virtual void deserialize(MPI_Status& stat);

    virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm);    

protected:

    std::vector<char> data;
    int size;    
};





