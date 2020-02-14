#pragma once

#include <mirheo/core/mirheo_object.h>

#include <mpi.h>
#include <vector>

namespace mirheo
{

class Simulation;

class Plugin
{    
public:
    Plugin();
    virtual ~Plugin();
    
    virtual void handshake();

    void setTag(int tag);
    
protected:
    MPI_Comm comm_, interComm_;
    int rank_, nranks_;
    
    void _setup(const MPI_Comm& comm, const MPI_Comm& interComm);

    int _sizeTag() const;
    int _dataTag() const;

private:
    void _checkTag() const;
    static constexpr int invalidTag = -1;
    int tag_ {invalidTag};
};

class SimulationPlugin : public Plugin, public MirSimulationObject
{
public:
    SimulationPlugin(const MirState *state, const std::string& name);
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
    void waitPrevSend();
    void send(const std::vector<char>& data);
    void send(const void *data, size_t sizeInBytes);
    ConfigObject _saveSnapshot(Saver&, const std::string& typeName);

private:
    int localSendSize_;
    MPI_Request sizeReq_, dataReq_;
};




class PostprocessPlugin : public Plugin, public MirObject
{
public:
    PostprocessPlugin(const std::string& name);
    virtual ~PostprocessPlugin();

    MPI_Request waitData();
    void recv();
    
    virtual void deserialize();

    virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm);    

protected:
    ConfigObject _saveSnapshot(Saver&, const std::string& typeName);

    std::vector<char> data_;
private:
    int size_;
};

// MirObject TypeLoadSave specialization works only if the derived class has
// MirObject as its first base class. The specialization cannot be easily fixed
// without including config.h from mirheo_object.h (to get ConfigValue
// definition), which would slow down the compilation. Instead, we specialize
// TypeLoadSave for plugins.
template <>
struct TypeLoadSave<SimulationPlugin>
{
    static ConfigValue save(Saver&, SimulationPlugin& obj);
};

template <>
struct TypeLoadSave<PostprocessPlugin>
{
    static ConfigValue save(Saver&, PostprocessPlugin& obj);
};

} // namespace mirheo
