#include "dump_mesh.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/celllist.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>

#include <regex>

MeshPlugin::MeshPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery) :
    SimulationPlugin(state, name), ovName(ovName),
    dumpEvery(dumpEvery)
{}

void MeshPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    ov = simulation->getOVbyNameOrDie(ovName);

    info("Plugin %s initialized for the following object vector: %s", name.c_str(), ovName.c_str());
}

void MeshPlugin::beforeForces(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    srcVerts = ov->local()->getMeshVertices(stream);
    srcVerts->downloadFromDevice(stream);
}

void MeshPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    debug2("Plugin %s is sending now data", name.c_str());

    vertices.clear();
    vertices.reserve(srcVerts->size());

    for (auto& p : *srcVerts)
        vertices.push_back(state->domain.local2global(make_float3(p)));

    auto& mesh = ov->mesh;

    MirState::StepType timeStamp = getTimeStamp(state, dumpEvery);
    
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, timeStamp, ov->name,
                                mesh->getNvertices(), mesh->getNtriangles(), mesh->triangles,
                                vertices);

    send(sendBuffer);
}

//=================================================================================

template<typename T>
static MPI_Offset writeToMPI(const std::vector<T>& data, MPI_File f, MPI_Offset base, MPI_Comm comm)
{    
    MPI_Offset offset = 0, nbytes = data.size()*sizeof(T);
    MPI_Check( MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, comm));

    MPI_Check( MPI_File_write_at_all(f, base + offset, data.data(), nbytes, MPI_CHAR, MPI_STATUS_IGNORE));

    MPI_Offset ntotal = 0;
    MPI_Check( MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, comm) );

    return ntotal;
}

static void writePLY(
        MPI_Comm comm, std::string fname,
        int nvertices, int nverticesPerObject,
        int ntriangles, int ntrianglesPerObject,
        int nObjects,
        const std::vector<int3>& mesh,
        const std::vector<float3>& vertices)
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

    int headerSize = 0;
    MPI_Offset fileOffset = 0;

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
        MPI_Check( MPI_File_write_at(f, fileOffset, content.c_str(), headerSize, MPI_CHAR, MPI_STATUS_IGNORE) );
    }

    MPI_Check( MPI_Bcast(&headerSize, 1, MPI_INT, 0, comm) );

    fileOffset += headerSize;
    
    fileOffset += writeToMPI(vertices, f, fileOffset, comm);

    int verticesOffset = 0;
    MPI_Check( MPI_Exscan(&nvertices, &verticesOffset, 1, MPI_INT, MPI_SUM, comm));

    std::vector<int4> connectivity;
    for(int j = 0; j < nObjects; ++j)
        for(int i = 0; i < ntrianglesPerObject; ++i)
        {
            int3 vertIds = mesh[i] + nverticesPerObject * j + verticesOffset;
            connectivity.push_back({3, vertIds.x, vertIds.y, vertIds.z});
        }

    fileOffset += writeToMPI(connectivity, f, fileOffset, comm);

    MPI_Check( MPI_File_close(&f));
}


MeshDumper::MeshDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(makePath(path))
{}

MeshDumper::~MeshDumper() = default;

void MeshDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void MeshDumper::deserialize(__UNUSED MPI_Status& stat)
{
    std::string ovName;
    int nvertices, ntriangles;

    MirState::StepType timeStamp;
    SimpleSerializer::deserialize(data, timeStamp, ovName, nvertices, ntriangles, connectivity, vertices);

    std::string currentFname = path + ovName + "_" + getStrZeroPadded(timeStamp) + ".ply";

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



