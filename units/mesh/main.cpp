#include <mirheo/core/logger.h>
#include <mirheo/core/mesh/mesh.h>
#include <mirheo/core/mesh/membrane.h>

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

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
