#include "membrane.h"

#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

MembraneMesh::MembraneMesh()
{}

MembraneMesh::MembraneMesh(std::string initialMesh) :
    Mesh(initialMesh)
{
    findAdjacent();
    computeInitialLengths(vertexCoordinates);
    computeInitialAreas(vertexCoordinates);
}

static bool sameFaces(const PinnedBuffer<int3>& facesA, const PinnedBuffer<int3>& facesB)
{
    if (facesA.size() != facesB.size())
        return false;
    
    for (int i = 0; i < facesA.size(); ++i) {
        int3 a = facesA[i];
        int3 b = facesB[i];

        if (a.x != b.x ||
            a.y != b.y ||
            a.z != b.z)
            return false;
    }

    return true;
}

MembraneMesh::MembraneMesh(std::string initialMesh, std::string stressFreeMesh) :
    Mesh(initialMesh)
{
    Mesh stressFree(stressFreeMesh);

    if (!sameFaces(triangles, stressFree.triangles))
        die("Must pass meshes with same connectivity for initial positions and stressFree vertices");
    
    if (vertexCoordinates.size() != stressFree.vertexCoordinates.size())
        die("Must pass same number of vertices for initial positions and stressFree vertices");
    
    findAdjacent();
    computeInitialLengths (stressFree.vertexCoordinates);
    computeInitialAreas   (stressFree.vertexCoordinates);
}

MembraneMesh::MembraneMesh(const PyTypes::VectorOfFloat3& vertices,
                           const PyTypes::VectorOfInt3& faces) :
    Mesh(vertices, faces)
{
    findAdjacent();
    computeInitialLengths(vertexCoordinates);
    computeInitialAreas(vertexCoordinates);
}

MembraneMesh::MembraneMesh(const PyTypes::VectorOfFloat3& vertices,
                           const PyTypes::VectorOfFloat3& stressFreeVertices,
                           const PyTypes::VectorOfInt3& faces) :
    Mesh(vertices, faces)
{
    if (vertices.size() != stressFreeVertices.size())
        die("Must pass same number of vertices for initial positions and stressFree vertices");
    
    Mesh stressFreeMesh(stressFreeVertices, faces);
    findAdjacent();
    computeInitialLengths(stressFreeMesh.vertexCoordinates);
    computeInitialAreas  (stressFreeMesh.vertexCoordinates);
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

void MembraneMesh::computeInitialLengths(const PinnedBuffer<float4>& vertices)
{
    initialLengths.resize_anew(nvertices * maxDegree);

    for (int i = 0; i < nvertices * maxDegree; i++) {
        if (adjacent[i] != NOT_SET)
            initialLengths[i] = length(vertices[i / maxDegree] - vertices[adjacent[i]]);
    }

    initialLengths.uploadToDevice(0);
}

static float computeArea(float3 v0, float3 v1, float3 v2) {
    return 0.5f * length(cross(v1 - v0, v2 - v0));
}

void MembraneMesh::computeInitialAreas(const PinnedBuffer<float4>& vertices)
{
    initialAreas.resize_anew(nvertices * maxDegree);

    float3 v0, v1, v2;

    for (int id0 = 0; id0 < nvertices; ++id0) {
        int degree = degrees[id0];
        int startId = id0 * maxDegree;
        v0 = f4tof3(vertices[id0]);
        
        for (int j = 0; j < degree; ++j) {
            int id1 = adjacent[startId + j];
            int id2 = adjacent[startId + (j + 1) % degree];

            assert(id2 != NOT_SET);

            v1 = f4tof3(vertices[id1]);
            v2 = f4tof3(vertices[id2]);

            initialAreas[startId + j] = computeArea(v0, v1, v2);
        }
    }

    initialAreas.uploadToDevice(0);
}
