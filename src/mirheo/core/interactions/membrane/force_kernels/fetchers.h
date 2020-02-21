#pragma once

#include "real.h"
#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

/// Fetch a vertex for a given view
class VertexFetcher
{
public:
    using VertexType = mReal3; ///< info contained in the fetched data
    using ViewType   = OVview; ///< compatible view

    /** \brief fetch a vertex coordinates from a view
        \param [in] view The view from which to fetch the vertex
        \param [in] i The index of the vertex in \p view
        \return The vertex coordinates
    */
    __D__ inline VertexType fetchVertex(const ViewType& view, int i) const
    {
        return make_mReal3(Real3_int(view.readPosition(i)).v);
    }
};

/// Fetch vertex coordinates and mean curvature for a given view
class VertexFetcherWithMeanCurvatures : public VertexFetcher
{
public:
    /// holds vertex coordinates and mean curvature
    struct VertexWithMeanCurvature
    {
        mReal3 r; ///< vertex coordinates
        mReal H;  ///< mean curvature
    };

    using VertexType = VertexWithMeanCurvature;   ///< info contained in the fetched data
    using ViewType   = OVviewWithJuelicherQuants; ///< compatible view
    
    /** \brief fetch a vertex coordinates and its mean curvature from a view
        \param [in] view The view from which to fetch the vertex
        \param [in] i The index of the vertex in \p view
        \return The vertex coordinates
    */
    __D__ inline VertexType fetchVertex(const ViewType& view, int i) const
    {
        return {VertexFetcher::fetchVertex(view, i),
                mReal(view.vertexMeanCurvatures[i])};
    }
};

} // namespace mirheo
