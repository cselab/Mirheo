#include "mesh.h"

#include <core/utils/cuda_common.h>

#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

Mesh::Mesh()
{}

Mesh::Mesh(std::string fname)
{
    _readOff(fname);
    _check();

    vertexCoordinates.uploadToDevice(0);
    triangles.uploadToDevice(0);

    _computeMaxDegree();
}

Mesh::Mesh(const PyTypes::VectorOfFloat3& vertices, const PyTypes::VectorOfInt3& faces)
{
    nvertices  = vertices.size();
    ntriangles = faces.size();

    vertexCoordinates.resize_anew(nvertices);
    triangles.resize_anew(ntriangles);

    for (int i = 0; i < ntriangles; ++i)
        triangles[i] = make_int3(faces[i][0], faces[i][1], faces[i][2]);

    for (int i = 0; i < nvertices; ++i)
        vertexCoordinates[i] = make_float4(vertices[i][0], vertices[i][1], vertices[i][2], 0.f);

    _check();
    
    vertexCoordinates.uploadToDevice(0);
    triangles.uploadToDevice(0);

    _computeMaxDegree();
}

Mesh::Mesh(Mesh&&) = default;

Mesh& Mesh::operator=(Mesh&&) = default;

Mesh::~Mesh() = default;

const int& Mesh::getNtriangles() const {return ntriangles;}
const int& Mesh::getNvertices()  const {return nvertices;}

const int& Mesh::getMaxDegree() const {
    if (maxDegree < 0) die("maxDegree was not computed");
    return maxDegree;
}

PyTypes::VectorOfFloat3 Mesh::getVertices()
{
    vertexCoordinates.downloadFromDevice(0, ContainersSynch::Synch);
    PyTypes::VectorOfFloat3 ret(getNvertices());

    for (int i = 0; i < getNvertices(); ++i) {
        auto r = vertexCoordinates[i];
        ret[i][0] = r.x;
        ret[i][1] = r.y;
        ret[i][2] = r.z;
    }
    return ret;
}

PyTypes::VectorOfInt3 Mesh::getTriangles()
{
    triangles.downloadFromDevice(0, ContainersSynch::Synch);
    PyTypes::VectorOfInt3 ret(getNtriangles());

    for (int i = 0; i < getNtriangles(); ++i) {
        auto t = triangles[i];
        ret[i][0] = t.x;
        ret[i][1] = t.y;
        ret[i][2] = t.z;
    }
    return ret;
}


void Mesh::_computeMaxDegree()
{
    std::vector<int> degrees(nvertices);

    for (auto t : triangles) {
        degrees[t.x] ++;
        degrees[t.y] ++;
        degrees[t.z] ++;
    }

    maxDegree = *std::max_element(degrees.begin(), degrees.end());
    debug("max degree is %d", maxDegree);
}

void Mesh::_check() const
{
    auto check = [this] (int tr) {
        if (tr < 0 || tr >= nvertices)
            die("Bad triangle indices");
    };

    for (int i = 0; i < getNtriangles(); ++i) {
        check(triangles[i].x);
        check(triangles[i].y);
        check(triangles[i].z);
    }
}

void Mesh::_readOff(std::string fname)
{
   std::ifstream fin(fname);
    if (!fin.good())
        die("Mesh file '%s' not found", fname.c_str());

    debug("Reading mesh from file '%s'", fname.c_str());

    std::string line;
    std::getline(fin, line); // OFF header

    int nedges;
    fin >> nvertices >> ntriangles >> nedges;
    std::getline(fin, line); // Finish with this line

    // Read the vertex coordinates
    vertexCoordinates.resize_anew(nvertices);
    for (int i=0; i<nvertices; i++)
        fin >> vertexCoordinates[i].x >> vertexCoordinates[i].y >> vertexCoordinates[i].z;

    // Read the connectivity data
    triangles.resize_anew(ntriangles);
    for (int i=0; i<ntriangles; i++)
    {
        int number;
        fin >> number;
        if (number != 3)
            die("Bad mesh file '%s' on line %d, number of face vertices is %d instead of 3",
                    fname.c_str(), 3 /* header */ + nvertices + i, number);

        fin >> triangles[i].x >> triangles[i].y >> triangles[i].z;
    }
}

MembraneMesh::MembraneMesh()
{}

MembraneMesh::MembraneMesh(std::string fname) : Mesh(fname)
{
    findAdjacent();
    computeInitialLengths();
    computeInitialAreas();
}

MembraneMesh::MembraneMesh(const PyTypes::VectorOfFloat3& vertices, const PyTypes::VectorOfInt3& faces) : Mesh(vertices, faces)
{
    findAdjacent();
    computeInitialLengths();
    computeInitialAreas();
}

MembraneMesh::MembraneMesh(MembraneMesh&&) = default;
MembraneMesh& MembraneMesh::operator=(MembraneMesh&&) = default;

MembraneMesh::~MembraneMesh() = default;

using EdgeMapPerVertex = std::vector< std::map<int, int> >;
static const int NOT_SET = -1;

static void findDegrees(const EdgeMapPerVertex& adjacentPairs, PinnedBuffer<int>& degrees)
{
    int nvertices = adjacentPairs.size();
    degrees.resize_anew(nvertices);
    
    for (int i = 0; i < nvertices; ++i)
        degrees[i] = adjacentPairs[i].size();
}

static void findNearestNeighbours(const EdgeMapPerVertex& adjacentPairs, int maxDegree, PinnedBuffer<int>& adjacent)
{
    int nvertices = adjacentPairs.size();

    adjacent.resize_anew(nvertices * maxDegree);
    std::fill(adjacent.begin(), adjacent.end(), NOT_SET);

    for (int v = 0; v < nvertices; ++v)
    {
        auto& l = adjacentPairs[v];
        auto myadjacent = &adjacent[maxDegree*v];
        
        // Add all the vertices on the adjacent edges one by one.
        myadjacent[0] = l.begin()->first;
        for (int i = 1; i < l.size(); ++i)
        {
            int current = myadjacent[i-1];
            
            assert(l.find(current) != l.end());
            myadjacent[i] =  l.find(current)->second;
        }
    }
}

void MembraneMesh::findAdjacent()
{
    // For every vertex: map from neigbouring vertex to a neigbour of both of vertices
    //
    //  all of such edges:
    //     V
    //
    //  <=====> 
    //   \   /
    //    \ /
    //     *
    EdgeMapPerVertex adjacentPairs(nvertices);

    for (const auto& t : triangles) {
        adjacentPairs [t.x][t.y] = t.z;
        adjacentPairs [t.y][t.z] = t.x;
        adjacentPairs [t.z][t.x] = t.y;
    }

    findDegrees(adjacentPairs, degrees);
    findNearestNeighbours(adjacentPairs, maxDegree, adjacent);
    
    adjacent.uploadToDevice(0);
    degrees.uploadToDevice(0);
}

void MembraneMesh::computeInitialLengths()
{
    initialLengths.resize_anew(nvertices * maxDegree);

    for (int i = 0; i < nvertices * maxDegree; i++) {
        if (adjacent[i] != NOT_SET)
            initialLengths[i] = length(vertexCoordinates[i / maxDegree] - vertexCoordinates[adjacent[i]]);
    }

    initialLengths.uploadToDevice(0);
}

static float computeArea(float3 v0, float3 v1, float3 v2) {
    return 0.5f * length(cross(v1 - v0, v2 - v0));
}

void MembraneMesh::computeInitialAreas()
{
    initialAreas.resize_anew(nvertices * maxDegree);

    float3 v0, v1, v2;

    for (int id0 = 0; id0 < nvertices; ++id0) {
        int degree = degrees[id0];
        int startId = id0 * maxDegree;
        v0 = f4tof3(vertexCoordinates[id0]);
        
        for (int j = 0; j < degree; ++j) {
            int id1 = adjacent[startId + j];
            int id2 = adjacent[startId + (j + 1) % degree];

            assert(id2 != NOT_SET);

            v1 = f4tof3(vertexCoordinates[id1]);
            v2 = f4tof3(vertexCoordinates[id2]);

            initialAreas[startId + j] = computeArea(v0, v1, v2);
        }
    }

    initialAreas.uploadToDevice(0);
}
