#include "edge_colors.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/cuda_common.h>

#include <algorithm>

namespace mirheo
{

std::vector<int> computeEdgeColors(const MembraneMesh *mesh)
{
    const int maxDegree = mesh->getMaxDegree();
    const int nv = mesh->getNvertices();

    const auto& adj = mesh->getAdjacents();
    const auto& deg = mesh->getDegrees();

    constexpr int noColor = -1;

    std::vector<int> colors(nv * maxDegree, noColor);

    auto usedColor = [&] (int i, int color) -> bool
    {
        for (int d = 0; d < deg[i]; ++d)
        {
            if (colors[maxDegree * i + d] == color)
                return true;
        }
        return false;
    };

    auto getOppositeAdjId = [&](int from, int to) -> int
    {
        for (int d = 0; d < deg[to]; ++d)
        {
            const int i = maxDegree * to + d;
            if (adj[i] == from)
                return i;
        }
        return -1;
    };

    // 2 * maxDegree is a conservative estimate
    for (int color = 0; color < 2*maxDegree; ++color)
    {
        for (int i = 0; i < nv; ++i)
        {
            for (int d = 0; d < deg[i]; ++d)
            {
                const int adjId = maxDegree * i + d;

                if (colors[adjId] == noColor &&
                    !usedColor(i, color) &&
                    !usedColor(adj[adjId], color))
                {
                    colors[adjId] = color;
                    colors[getOppositeAdjId(i, adj[adjId])] = color;
                    break;
                }
            }
        }
    }
    return colors;
}


MeshDistinctEdgeSets::MeshDistinctEdgeSets(const MembraneMesh *mesh)
{
    const auto colors = computeEdgeColors(mesh);
    const int maxColor = *std::max_element(colors.begin(), colors.end());
    const int numColors = maxColor + 1;

    std::vector<std::vector<int2>> edges(numColors);

    const int nv = mesh->getNvertices();
    const int md = mesh->getMaxDegree();
    const auto& deg = mesh->getDegrees();
    const auto& adj = mesh->getAdjacents();

    for (int i = 0; i < nv; ++i)
    {
        for (int d = 0; d < deg[i]; ++d)
        {
            const int j = adj[i * md + d];

            // store the edges only once.
            if (j < i)
                continue;

            const int color = colors[i * md + d];
            edges[color].push_back(int2{i, j});
        }
    }

    edges_.resize(numColors);

    for (int c = 0; c < numColors; ++c)
    {
        edges_[c].resize_anew(edges[c].size());
        std::copy(edges[c].begin(), edges[c].end(), edges_[c].begin());
        edges_[c].uploadToDevice(defaultStream);
    }
}

int MeshDistinctEdgeSets::numColors() const
{
    return (int) edges_.size();
}

const PinnedBuffer<int2>& MeshDistinctEdgeSets::edgeSet(int color) const
{
    if (color < 0 || color >= numColors())
        die("asked for color %d, but have only %d colors.",
            color, numColors());
    return edges_[color];
}


} // namespace mirheo
