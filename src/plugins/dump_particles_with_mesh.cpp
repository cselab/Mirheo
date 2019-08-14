#include "dump_particles_with_mesh.h"
#include "utils/simple_serializer.h"

#include <core/simulation.h>
#include <core/utils/folders.h>

ParticleWithMeshSenderPlugin::ParticleWithMeshSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                                                           std::vector<std::string> channelNames,
                                                           std::vector<ChannelType> channelTypes) :
    ParticleSenderPlugin(state, name, pvName, dumpEvery, channelNames, channelTypes)
{}

void ParticleWithMeshSenderPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getOVbyNameOrDie(pvName);

    info("Plugin %s initialized for the following object vector: %s", name.c_str(), pvName.c_str());
}

void ParticleWithMeshSenderPlugin::handshake()
{
    ParticleSenderPlugin::handshake();

    auto& mesh = static_cast<ObjectVector*>(pv)->mesh;

    waitPrevSend();
    debug("handshake for plugin '%s': sending %d triangles for a %d vertices mesh", name.c_str(), mesh->getNtriangles(), mesh->getNvertices());
    SimpleSerializer::serialize(sendBuffer, mesh->getNvertices(), mesh->triangles);
    send(sendBuffer);
}




ParticleWithMeshDumperPlugin::ParticleWithMeshDumperPlugin(std::string name, std::string path) :
    ParticleDumperPlugin(name, path),
    allTriangles(std::make_shared<std::vector<int3>>())
{}

void ParticleWithMeshDumperPlugin::handshake()
{
    ParticleDumperPlugin::handshake();

    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    SimpleSerializer::deserialize(data, nvertices, triangles);
    debug("handshake for plugin '%s': received %d triangles for a %d vertices mesh", name.c_str(), triangles.size(), nvertices);
}

void ParticleWithMeshDumperPlugin::_prepareConnectivity(int totNVertices)
{
    if (totNVertices % nvertices != 0)
        die("plugin '%s' expecting a multiple of %d vertices, got %d", name.c_str(), nvertices, totNVertices);

    long nobjects = totNVertices / nvertices;
    long offset   = 0;

    int ntriangles = triangles.size();
    
    MPI_Check( MPI_Exscan(&nobjects, &offset, 1, MPI_LONG, MPI_SUM, comm) );

    allTriangles->resize(nobjects * ntriangles);

    auto *connectivity = allTriangles->data();
    
    for (int i = 0; i < nobjects; ++i) {
        int start = nvertices * (offset + i);        
        for (int j = 0; j < ntriangles; ++j) {
            int id = i * ntriangles + j;
            int3 t = triangles[j];
            
            connectivity[id] = start + t;
        }
    }
}

void ParticleWithMeshDumperPlugin::deserialize(__UNUSED MPI_Status& stat)
{
    debug2("Plugin '%s' will dump right now", name.c_str());

    MirState::TimeType time;
    MirState::StepType timeStamp;
    _recvAndUnpack(time, timeStamp);
    
    int totNVertices = positions->size();    

    _prepareConnectivity(totNVertices);

    std::string fname = path + getStrZeroPadded(timeStamp, zeroPadding);
    
    XDMF::TriangleMeshGrid grid(positions, allTriangles, comm);
    XDMF::write(fname, &grid, channels, time, comm);
}
