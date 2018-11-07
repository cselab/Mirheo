#include <core/simulation.h>
#include <core/utils/folders.h>
#include "simple_serializer.h"


#include "dump_particles_with_mesh.h"

ParticleWithMeshSenderPlugin::ParticleWithMeshSenderPlugin(std::string name, std::string pvName, int dumpEvery,
                                                           std::vector<std::string> channelNames,
                                                           std::vector<ChannelType> channelTypes) :
    ParticleSenderPlugin(name, pvName, dumpEvery, channelNames, channelTypes)
{}

void ParticleWithMeshSenderPlugin::setup(Simulation *sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(sim, comm, interComm);

    pv = sim->getOVbyNameOrDie(pvName);

    info("Plugin %s initialized for the following object vector: %s", name().c_str(), pvName.c_str());
}

void ParticleWithMeshSenderPlugin::handshake()
{
    ParticleSenderPlugin::handshake();

    auto& mesh = static_cast<ObjectVector*>(pv)->mesh;

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, mesh->getNvertices(), mesh->triangles);
    send(sendBuffer);
}




ParticleWithMeshDumperPlugin::ParticleWithMeshDumperPlugin(std::string name, std::string path) :
    ParticleDumperPlugin(name, path), allTriangles(new std::vector<int>())
{}

void ParticleWithMeshDumperPlugin::handshake()
{
    ParticleDumperPlugin::handshake();

    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    SimpleSerializer::deserialize(data, nvertices, triangles);
}

void ParticleWithMeshDumperPlugin::_prepareConnectivity(int totNVertices)
{
    if (totNVertices % nvertices != 0)
        die("plugin '%s' expecting a multiple of %d vertices, got %d", name().c_str(), nvertices, totNVertices);

    long nobjects = totNVertices / nvertices;
    long offset   = 0;

    int ntriangles = triangles.size();
    
    MPI_Check( MPI_Exscan(&nobjects, &offset, 1, MPI_LONG, MPI_SUM, comm) );

    allTriangles->resize(nobjects * 3 * ntriangles);

    auto *connectivity = allTriangles->data();
    
    for (int i = 0; i < nobjects; ++i) {
        int start = nvertices * (offset + i);        
        for (int j = 0; j < ntriangles; ++j) {
            int id = i * ntriangles + j;
            int3 t = triangles[j];
            
            connectivity[3 * id + 0] = start + t.x;
            connectivity[3 * id + 1] = start + t.y;
            connectivity[3 * id + 2] = start + t.z;
        }
    }
}

void ParticleWithMeshDumperPlugin::deserialize(MPI_Status& stat)
{
    debug2("Plugin '%s' will dump right now", name().c_str());

    float t = _recvAndUnpack();

    int totNVertices = positions->size() / 3;    

    _prepareConnectivity(totNVertices);

    std::string fname = path + getStrZeroPadded(timeStamp++, zeroPadding);
    
    XDMF::TriangleMeshGrid grid(positions, allTriangles, comm);
    XDMF::write(fname, &grid, channels, t, comm);
}
