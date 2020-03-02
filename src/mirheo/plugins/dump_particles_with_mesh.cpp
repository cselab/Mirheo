#include "dump_particles_with_mesh.h"
#include "utils/simple_serializer.h"

#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/folders.h>

namespace mirheo
{

ParticleWithMeshSenderPlugin::ParticleWithMeshSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                                                           const std::vector<std::string>& channelNames) :
    ParticleSenderPlugin(state, name, pvName, dumpEvery, channelNames)
{}

void ParticleWithMeshSenderPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getOVbyNameOrDie(pvName_);

    info("Plugin %s initialized for the following object vector: %s", getCName(), pvName_.c_str());
}

void ParticleWithMeshSenderPlugin::handshake()
{
    ParticleSenderPlugin::handshake();

    auto& mesh = static_cast<ObjectVector*>(pv_)->mesh;

    waitPrevSend();
    debug("handshake for plugin '%s': sending %d triangles for a %d vertices mesh", getCName(), mesh->getNtriangles(), mesh->getNvertices());
    SimpleSerializer::serialize(sendBuffer_, mesh->getNvertices(), mesh->getFaces());
    send(sendBuffer_);
}




ParticleWithMeshDumperPlugin::ParticleWithMeshDumperPlugin(std::string name, std::string path) :
    ParticleDumperPlugin(name, path),
    allTriangles_(std::make_shared<std::vector<int3>>())
{}

void ParticleWithMeshDumperPlugin::handshake()
{
    ParticleDumperPlugin::handshake();

    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    SimpleSerializer::deserialize(data_, nvertices_, triangles_);
    debug("handshake for plugin '%s': received %zu triangles for a %d vertices mesh",
          getCName(), triangles_.size(), nvertices_);
}

void ParticleWithMeshDumperPlugin::_prepareConnectivity(int totNVertices)
{
    if (totNVertices % nvertices_ != 0)
        die("plugin '%s' expecting a multiple of %d vertices, got %d", getCName(), nvertices_, totNVertices);

    const int nobjects = totNVertices / nvertices_;
    int offset   = 0;

    const int ntriangles = static_cast<int>(triangles_.size());
    
    MPI_Check( MPI_Exscan(&nobjects, &offset, 1, MPI_INT, MPI_SUM, comm_) );

    allTriangles_->resize(nobjects * ntriangles);

    auto *connectivity = allTriangles_->data();
    
    for (int i = 0; i < nobjects; ++i)
    {
        const int start = nvertices_ * (offset + i);        
        for (int j = 0; j < ntriangles; ++j)
        {
            const int id = i * ntriangles + j;
            const int3 t = triangles_[j];
            connectivity[id] = start + t;
        }
    }
}

void ParticleWithMeshDumperPlugin::deserialize()
{
    debug2("Plugin '%s' will dump right now", getCName());

    MirState::TimeType time;
    MirState::StepType timeStamp;
    _recvAndUnpack(time, timeStamp);
    
    const int totNVertices = static_cast<int>(positions_->size());    

    _prepareConnectivity(totNVertices);

    const std::string fname = path_ + getStrZeroPadded(timeStamp, zeroPadding_);
    
    const XDMF::TriangleMeshGrid grid(positions_, allTriangles_, comm_);
    XDMF::write(fname, &grid, channels_, time, comm_);
}

} // namespace mirheo
