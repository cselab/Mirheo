#include "dump_mesh.h"
#include "simple_serializer.h"
#include <core/utils/folders.h>

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>

#include <regex>

MeshPlugin::MeshPlugin(std::string name, std::string ovName, int dumpEvery) :
SimulationPlugin(name), ovName(ovName),
dumpEvery(dumpEvery)
{ }

void MeshPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(sim, comm, interComm);

    ov = sim->getOVbyNameOrDie(ovName);

    info("Plugin %s initialized for the following object vector: %s", name.c_str(), ovName.c_str());
}

void MeshPlugin::beforeForces(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

    srcVerts = ov->local()->getMeshVertices(stream);
    srcVerts->downloadFromDevice(stream);
}

void MeshPlugin::serializeAndSend(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

    debug2("Plugin %s is sending now data", name.c_str());

    vertices.clear();
    vertices.reserve(srcVerts->size());

    for (auto& p : *srcVerts)
        vertices.push_back(ov->domain.local2global(p.r));

    auto& mesh = ov->mesh;

    waitPrevSend();
    SimpleSerializer::serialize(data, ov->name,
            mesh->getNvertices(), mesh->getNtriangles(), mesh->triangles,
            vertices);

    send(data);
}

//=================================================================================

template<typename T>
void writeToMPI(const std::vector<T> data, MPI_File f, MPI_Comm comm)
{
    MPI_Offset base;
    MPI_Check( MPI_File_get_position(f, &base));

    MPI_Offset offset = 0, nbytes = data.size()*sizeof(T);
    MPI_Check( MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, comm));

    MPI_Check( MPI_File_write_at_all(f, base + offset, data.data(), nbytes, MPI_CHAR, MPI_STATUS_IGNORE));

    MPI_Offset ntotal = 0;
    MPI_Check( MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, comm) );

    MPI_Check( MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}

void writePLY(
        MPI_Comm comm, std::string fname,
        int nvertices, int nverticesPerObject,
        int ntriangles, int ntrianglesPerObject,
        int nObjects,
        std::vector<int3>& mesh,
        std::vector<float3>& vertices)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    int totalVerts = 0;
    MPI_Check( MPI_Reduce(&nvertices, &totalVerts, 1, MPI_INT, MPI_SUM, 0, comm) );

    int totalTriangles = 0;
    MPI_Check( MPI_Reduce(&ntriangles, &totalTriangles, 1, MPI_INT, MPI_SUM, 0, comm) );

    MPI_File f;
    MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE|MPI_MODE_DELETE_ON_CLOSE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );
    MPI_Check( MPI_File_close(&f) );
    MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f) );

    int headerSize;

    if (rank == 0)
    {
        std::stringstream ss;

        ss <<  "ply\n";
        ss <<  "format binary_little_endian 1.0\n";
        ss <<  "element vertex " << totalVerts << "\n";
        ss <<  "property float x\nproperty float y\nproperty float z\n";
        //ss <<  "property float xnormal\nproperty float ynormal\nproperty float znormal\n";
        ss <<  "element face " << totalTriangles << "\n";
        ss <<  "property list int int vertex_index\n";
        ss <<  "end_header\n";

        std::string content = ss.str();
        headerSize = content.length();
        MPI_Check( MPI_File_write_at(f, 0, content.c_str(), headerSize, MPI_CHAR, MPI_STATUS_IGNORE) );
    }

    MPI_Check( MPI_Bcast(&headerSize, 1, MPI_INT, 0, comm) );
    MPI_Check( MPI_File_seek(f, headerSize, MPI_SEEK_CUR));


    writeToMPI(vertices, f, comm);

    int verticesOffset = 0;
    MPI_Check( MPI_Exscan(&nvertices, &verticesOffset, 1, MPI_INT, MPI_SUM, comm));

    std::vector<int4> connectivity;
    for(int j = 0; j < nObjects; ++j)
        for(int i = 0; i < ntrianglesPerObject; ++i)
        {
            int3 vertIds = mesh[i] + nverticesPerObject * j + verticesOffset;
            connectivity.push_back({3, vertIds.x, vertIds.y, vertIds.z});
        }

    writeToMPI(connectivity, f, comm);

    MPI_Check( MPI_File_close(&f));
}


MeshDumper::MeshDumper(std::string name, std::string path) :
                PostprocessPlugin(name), path(path)
{    }

void MeshDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void MeshDumper::deserialize(MPI_Status& stat)
{
    std::string ovName;
    int nvertices, ntriangles;

    SimpleSerializer::deserialize(data, ovName, nvertices, ntriangles, connectivity, vertices);

    std::string tstr = std::to_string(timeStamp++);
    std::string currentFname = path + "/" + ovName + "_" + std::string(5 - tstr.length(), '0') + tstr + ".ply";

    if (activated)
    {
        int nObjects = vertices.size() / nvertices;
        writePLY(comm, currentFname,
                nvertices*nObjects, nvertices,
                ntriangles*nObjects, ntriangles,
                nObjects,
                connectivity, vertices);
    }
}



