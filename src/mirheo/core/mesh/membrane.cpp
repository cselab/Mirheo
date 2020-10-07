// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "membrane.h"

#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/path.h>

#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

namespace mirheo
{

/// Allocate and read a buffer of reals.
static void readReals(FILE *f, PinnedBuffer<real> *buffer)
{
    size_t size;
    if (1 != fscanf(f, "%zu", &size))
        die("fscanf error");
    buffer->resize_anew(size);
    for (size_t i = 0; i < buffer->size(); ++i)
        if (1 != fscanf(f, "%" MIRHEO_SCNgREAL, &(*buffer)[i]))
            die("fscanf error");
    buffer->uploadToDevice(defaultStream);
}

/// Print a buffer of reals to a given file.
static void writeReals(FILE *f, const PinnedBuffer<real> &buffer)
{
    fprintf(f, "%zu\n", buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) {
        fprintf(f, "%*" MIRHEO_PRIgREAL "\n",
                std::numeric_limits<real>::max_digits10, buffer[i]);
    }
}


MembraneMesh::MembraneMesh()
{}

MembraneMesh::MembraneMesh(const std::string& initialMesh) :
    Mesh(initialMesh)
{
    _findAdjacent();
    _computeInitialQuantities(vertices_);
}



static bool sameFaces(const PinnedBuffer<int3>& facesA, const PinnedBuffer<int3>& facesB)
{
    if (facesA.size() != facesB.size())
        return false;

    for (size_t i = 0; i < facesA.size(); ++i)
    {
        int3 a = facesA[i];
        int3 b = facesB[i];

        if (a.x != b.x ||
            a.y != b.y ||
            a.z != b.z)
            return false;
    }

    return true;
}

MembraneMesh::MembraneMesh(const std::string& initialMesh, const std::string& stressFreeMesh) :
    Mesh(initialMesh)
{
    Mesh stressFree(stressFreeMesh);

    if (!sameFaces(this->getFaces(), stressFree.getFaces()))
        die("Must pass meshes with same connectivity for initial positions and stressFree vertices");

    if (this->getNvertices() != stressFree.getNvertices())
        die("Must pass same number of vertices for initial positions and stressFree vertices");

    _findAdjacent();
    _computeInitialQuantities(stressFree.getVertices());
}

MembraneMesh::MembraneMesh(const std::vector<real3>& vertices,
                           const std::vector<int3>& faces) :
    Mesh(vertices, faces)
{
    _findAdjacent();
    _computeInitialQuantities(vertices_);
}

MembraneMesh::MembraneMesh(const std::vector<real3>& vertices,
                           const std::vector<real3>& stressFreeVertices,
                           const std::vector<int3>& faces) :
    Mesh(vertices, faces)
{
    if (vertices.size() != stressFreeVertices.size())
        die("Must pass same number of vertices for initial positions and stressFree vertices");

    Mesh stressFreeMesh(stressFreeVertices, faces);
    _findAdjacent();
    _computeInitialQuantities(stressFreeMesh.getVertices());
}

MembraneMesh::MembraneMesh(Loader& loader, const ConfigObject& config) :
    Mesh(loader, config)
{
    _findAdjacent();
    // Replacement for _computeInitialQuantities(stressFreeMesh.vertexCoordinates).
    std::string fileName = joinPaths(loader.getContext().getPath(), config["name"] + ".stressFree.dat");
    FileWrapper f(fileName, "r");
    readReals(f.get(), &initialLengths_);
    readReals(f.get(), &initialAreas_);
    readReals(f.get(), &initialDotProducts_);
}


MembraneMesh::MembraneMesh(MembraneMesh&&) = default;
MembraneMesh& MembraneMesh::operator=(MembraneMesh&&) = default;

MembraneMesh::~MembraneMesh() = default;

void MembraneMesh::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<MembraneMesh>(this, _saveSnapshot(saver, "MembraneMesh"));
}

ConfigObject MembraneMesh::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = Mesh::_saveSnapshot(saver, typeName);

    std::string fileName = joinPaths(saver.getContext().path, config["name"] + ".stressFree.dat");
    FileWrapper f(fileName, "w");
    writeReals(f.get(), initialLengths_);
    writeReals(f.get(), initialAreas_);
    writeReals(f.get(), initialDotProducts_);
    return config;
}

using EdgeMapPerVertex = std::vector< std::map<int, int> >;
constexpr int invalidId = -1;

static void findDegrees(const EdgeMapPerVertex& adjacentPairs, PinnedBuffer<int>& degrees)
{
    const size_t nvertices = adjacentPairs.size();
    degrees.resize_anew(nvertices);

    for (size_t i = 0; i < nvertices; ++i)
        degrees[i] = static_cast<int>(adjacentPairs[i].size());
}

static void findNearestNeighbours(const EdgeMapPerVertex& adjacentPairs, int maxDegree, PinnedBuffer<int>& adjacent)
{
    const size_t nvertices = adjacentPairs.size();

    adjacent.resize_anew(nvertices * maxDegree);
    std::fill(adjacent.begin(), adjacent.end(), invalidId);

    for (size_t v = 0; v < nvertices; ++v)
    {
        auto& l = adjacentPairs[v];
        auto myadjacent = &adjacent[maxDegree*v];

        // Add all the vertices on the adjacent edges one by one.
        myadjacent[0] = l.begin()->first;
        for (size_t i = 1; i < l.size(); ++i)
        {
            const int current = myadjacent[i-1];

            if (l.find(current) == l.end())
                die("Unexpected adjacent pairs. This might come from a bad connectivity of the input mesh");

            myadjacent[i] = l.find(current)->second;
        }
    }
}

void MembraneMesh::_findAdjacent()
{
    /*
     For every vertex: map from neigbouring vertex to a neigbour of both of vertices

      all of such edges:
         V

      <=====>
       \   /
        \ /
         *
    */
    EdgeMapPerVertex adjacentPairs(getNvertices());

    for (const auto& t : faces_)
    {
        adjacentPairs [t.x][t.y] = t.z;
        adjacentPairs [t.y][t.z] = t.x;
        adjacentPairs [t.z][t.x] = t.y;
    }

    findDegrees(adjacentPairs, degrees_);
    findNearestNeighbours(adjacentPairs, getMaxDegree(), adjacent_);

    adjacent_.uploadToDevice(defaultStream);
    degrees_.uploadToDevice(defaultStream);
}

void MembraneMesh::_computeInitialQuantities(const PinnedBuffer<real4>& vertices)
{
    _computeInitialLengths(vertices);
    _computeInitialAreas(vertices);
    _computeInitialDotProducts(vertices);
}

void MembraneMesh::_computeInitialLengths(const PinnedBuffer<real4>& vertices)
{
    initialLengths_.resize_anew(getNvertices() * getMaxDegree());

    for (int i = 0; i < getNvertices() * getMaxDegree(); i++)
    {
        if (adjacent_[i] != invalidId)
            initialLengths_[i] = length(vertices[i / getMaxDegree()] - vertices[adjacent_[i]]);
    }

    initialLengths_.uploadToDevice(defaultStream);
}

static real computeArea(real3 v0, real3 v1, real3 v2)
{
    return 0.5_r * length(cross(v1 - v0, v2 - v0));
}

void MembraneMesh::_computeInitialAreas(const PinnedBuffer<real4>& vertices)
{
    initialAreas_.resize_anew(getNvertices() * getMaxDegree());

    real3 v0, v1, v2;

    for (int id0 = 0; id0 < getNvertices(); ++id0)
    {
        const int degree = degrees_
            [id0];
        const int startId = id0 * getMaxDegree();
        v0 = make_real3(vertices[id0]);

        for (int j = 0; j < degree; ++j)
        {
            const int id1 = adjacent_[startId + j];
            const int id2 = adjacent_[startId + (j + 1) % degree];

            v1 = make_real3(vertices[id1]);
            v2 = make_real3(vertices[id2]);

            initialAreas_[startId + j] = computeArea(v0, v1, v2);
        }
    }

    initialAreas_.uploadToDevice(defaultStream);
}

void MembraneMesh::_computeInitialDotProducts(const PinnedBuffer<real4>& vertices)
{
    initialDotProducts_.resize_anew(getNvertices() * getMaxDegree());

    real3 v0, v1, v2;

    for (int id0 = 0; id0 < getNvertices(); ++id0)
    {
        const int degree = degrees_[id0];
        const int startId = id0 * getMaxDegree();
        v0 = make_real3(vertices[id0]);

        for (int j = 0; j < degree; ++j)
        {
            const int id1 = adjacent_[startId + j];
            const int id2 = adjacent_[startId + (j + 1) % degree];

            v1 = make_real3(vertices[id1]);
            v2 = make_real3(vertices[id2]);

            initialDotProducts_[startId + j] = dot(v1 - v0, v2 - v0);
        }
    }

    initialDotProducts_.uploadToDevice(defaultStream);
}


MembraneMeshView::MembraneMeshView(const MembraneMesh *m) :
    MeshView(m),
    maxDegree          (m->getMaxDegree()),
    adjacent           (m->adjacent_.devPtr()),
    degrees            (m->degrees_.devPtr()),
    initialLengths     (m->initialLengths_.devPtr()),
    initialAreas       (m->initialAreas_.devPtr()),
    initialDotProducts (m->initialDotProducts_.devPtr())
{}

} // namespace mirheo
