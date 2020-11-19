// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "membrane.h"

#include <vector>

namespace mirheo
{

/** Color each edge of the given mesh such that each vertex has at most one edge with a given color.
    \param [in] mesh The mesh that contains adjacency information.
    \return The color of all edges, in the adjacency list order.
 */
std::vector<int> computeEdgeColors(const MembraneMesh *mesh);


/** Stores sets of edges that share the same colors as computed by computeEdgeColors().
    This allows to work on edges in parallel with no race conditions.
 */
class MeshDistinctEdgeSets
{
public:
    /** Construct a MeshDistinctEdgeSets.
        \param [in] The input mesh with adjacency lists.
    */
    MeshDistinctEdgeSets(const MembraneMesh *mesh);

    /// \return the number of colors in the associated mesh.
    int numColors() const;

    /// \return The list of edges (vertex indices pairs) that have the given color.
    const PinnedBuffer<int2>& edgeSet(int color) const;

private:
    std::vector<PinnedBuffer<int2>> edges_;
};

} // namespace mirheo
