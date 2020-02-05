#include "dump_mesh.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/folders.h>

#include <regex>

namespace mirheo
{

MeshPlugin::MeshPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery) :
    SimulationPlugin(state, name),
    ovName_(ovName),
    dumpEvery_(dumpEvery)
{}

void MeshPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    ov_ = simulation->getOVbyNameOrDie(ovName_);

    info("Plugin %s initialized for the following object vector: %s", getCName(), ovName_.c_str());
}

void MeshPlugin::beforeForces(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    srcVerts_ = ov_->local()->getMeshVertices(stream);
    srcVerts_->downloadFromDevice(stream);
}

void MeshPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    debug2("Plugin %s is sending now data", getCName());

    vertices_.clear();
    vertices_.reserve(srcVerts_->size());

    for (auto& p : *srcVerts_)
        vertices_.push_back(getState()->domain.local2global(make_real3(p)));

    auto& mesh = ov_->mesh;

    MirState::StepType timeStamp = getTimeStamp(getState(), dumpEvery_);
    
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, timeStamp, ov_->getName(),
                                mesh->getNvertices(), mesh->getNtriangles(), mesh->triangles,
                                vertices_);

    send(sendBuffer_);
}

//=================================================================================

template<typename T>
static MPI_Offset writeToMPI(const std::vector<T>& data, MPI_File f, MPI_Offset base, MPI_Comm comm)
{    
    MPI_Offset offset = 0;
    const MPI_Offset nbytes = data.size() * sizeof(T);
    MPI_Check( MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, comm));

    MPI_Check( MPI_File_write_at_all(f, base + offset, data.data(), static_cast<int>(nbytes), MPI_BYTE, MPI_STATUS_IGNORE));

    MPI_Offset ntotal = 0;
    MPI_Check( MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, comm) );

    return ntotal;
}

template<typename T> inline std::string getTypeStr();
template<> inline std::string getTypeStr<float> () {return "float";}
template<> inline std::string getTypeStr<double>() {return "double";}

static void writePLY(
        MPI_Comm comm, std::string fname,
        int nvertices, int nverticesPerObject,
        int ntriangles, int ntrianglesPerObject,
        int nObjects,
        const std::vector<int3>& mesh,
        const std::vector<real3>& vertices)
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
        const std::string vertexTypeStr = getTypeStr<real>();

        ss <<  "ply\n";
        ss <<  "format binary_little_endian 1.0\n";
        ss <<  "element vertex " << totalVerts << "\n";
        ss <<  "property " << vertexTypeStr << " x\n";
        ss <<  "property " << vertexTypeStr << " y\n";
        ss <<  "property " << vertexTypeStr << " z\n";
        ss <<  "element face " << totalTriangles << "\n";
        ss <<  "property list int int vertex_index\n";
        ss <<  "end_header\n";

        const std::string content = ss.str();
        headerSize = static_cast<int>(content.length());
        MPI_Check( MPI_File_write_at(f, fileOffset, content.c_str(), headerSize, MPI_CHAR, MPI_STATUS_IGNORE) );
    }

    MPI_Check( MPI_Bcast(&headerSize, 1, MPI_INT, 0, comm) );

    fileOffset += headerSize;
    
    fileOffset += writeToMPI(vertices, f, fileOffset, comm);

    int verticesOffset = 0;
    MPI_Check( MPI_Exscan(&nvertices, &verticesOffset, 1, MPI_INT, MPI_SUM, comm));

    std::vector<int4> connectivity;
    for (int j = 0; j < nObjects; ++j)
        for (int i = 0; i < ntrianglesPerObject; ++i)
        {
            const int3 vertIds = mesh[i] + nverticesPerObject * j + verticesOffset;
            connectivity.push_back({3, vertIds.x, vertIds.y, vertIds.z});
        }

    fileOffset += writeToMPI(connectivity, f, fileOffset, comm);

    MPI_Check( MPI_File_close(&f));
}


MeshDumper::MeshDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

MeshDumper::~MeshDumper() = default;

void MeshDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void MeshDumper::deserialize()
{
    std::string ovName;
    int nvertices, ntriangles;

    MirState::StepType timeStamp;
    SimpleSerializer::deserialize(data, timeStamp, ovName, nvertices, ntriangles, connectivity_, vertices_);

    std::string currentFname = path_ + ovName + "_" + getStrZeroPadded(timeStamp) + ".ply";

    if (activated_)
    {
        const int nObjects = static_cast<int>(vertices_.size()) / nvertices;
        writePLY(comm, currentFname,
                nvertices * nObjects, nvertices,
                ntriangles*nObjects, ntriangles,
                nObjects,
                connectivity_, vertices_);
    }
}

} // namespace mirheo
