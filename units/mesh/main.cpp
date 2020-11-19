#include <mirheo/core/logger.h>
#include <mirheo/core/mesh/mesh.h>
#include <mirheo/core/mesh/membrane.h>
#include <mirheo/core/mesh/edge_colors.h>

#include <cstdio>
#include <cmath>
#include <set>
#include <string>
#include <gtest/gtest.h>

using namespace mirheo;

static const std::string rbc_off = "../../data/rbc_mesh.off";

TEST (MESH, readOff)
{
    Mesh mesh(rbc_off);
    ASSERT_EQ(mesh.getNtriangles(), 992);
    ASSERT_EQ(mesh.getNvertices(), 498);
    ASSERT_EQ(mesh.getMaxDegree(), 7);
}

TEST (MESH, adjacencyList)
{
    MembraneMesh mesh(rbc_off);
    const int nv = mesh.getNvertices();
    const int md = mesh.getMaxDegree();
    const auto& adj = mesh.getAdjacents();
    const auto& deg = mesh.getDegrees();

    std::set<std::pair<int, int>> edges;

    for (int i = 0; i < nv; ++i)
    {
        for (int d = 0; d < deg[i]; ++d)
        {
            const int j = adj[i * md + d];
            ASSERT_LT(j, nv);
            ASSERT_GE(j, 0);
            edges.insert({i, j});
        }
    }

    // test symmetry
    for (int i = 0; i < nv; ++i)
    {
        for (int d = 0; d < deg[i]; ++d)
        {
            const int j = adj[i * md + d];
            ASSERT_NE(edges.find({j, i}), edges.end());
        }
    }
}

static void checkColors(const MembraneMesh *mesh)
{
    const auto colors = computeEdgeColors(mesh);

    const int nv = mesh->getNvertices();
    const int maxDegree = mesh->getMaxDegree();
    const auto& deg = mesh->getDegrees();

    // check every edge is assigned to the at least one color
    for (int i = 0; i < nv; ++i)
    {
        for (int d = 0; d < deg[i]; ++d)
        {
            const int color = colors[maxDegree * i + d];
            // printf("%d: %d\n", i, color);
            ASSERT_NE(color, -1);
        }
    }

    // check that every vertex has each color once at most
    std::vector<std::vector<int>> vertexToColors(nv);
    for (int i = 0; i < nv; ++i)
    {
        auto& myColors = vertexToColors[i];

        for (int d = 0; d < deg[i]; ++d)
        {
            const int color = colors[maxDegree * i + d];
            ASSERT_EQ(std::find(myColors.begin(), myColors.end(), color), myColors.end());
            myColors.push_back(color);
        }
    }
}

TEST (MESH, edgeColors)
{
    MembraneMesh mesh(rbc_off);
    checkColors(&mesh);
}

TEST (MESH, edgeSets)
{
    MembraneMesh mesh(rbc_off);
    MeshDistinctEdgeSets edgeSets(&mesh);

    std::set<int> allUsedVertices;
    int numUsedEdges{0};

    // check there is no overlapping vertices
    for (int color = 0; color < edgeSets.numColors(); ++color)
    {
        std::set<int> usedVertices;
        for (const auto edge : edgeSets.edgeSet(color))
        {
            ASSERT_EQ(usedVertices.find(edge.x), usedVertices.end());
            usedVertices.insert(edge.x);

            ASSERT_EQ(usedVertices.find(edge.y), usedVertices.end());
            usedVertices.insert(edge.y);

            allUsedVertices.insert(edge.x);
            allUsedVertices.insert(edge.y);
            ++numUsedEdges;
        }
    }

    // check that we used all vertices
    ASSERT_EQ((int) allUsedVertices.size(), mesh.getNvertices());

    // Euler Formula assuming the mesh has genus 0
    const int numEdges = mesh.getNvertices() + mesh.getNtriangles() - 2;
    ASSERT_EQ(numUsedEdges, numEdges);
}




int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
