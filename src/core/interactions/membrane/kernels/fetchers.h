#pragma once

#include "real.h"
#include <core/utils/cpu_gpu_defines.h>

class VertexFetcher
{
public:
    using VertexType = mReal3;
    using ViewType   = OVview;

    __D__ inline VertexType fetchVertex(const ViewType& view, int i) const
    {
        return make_mReal3(Real3_int(view.readPosition(i)).v);
    }
};

class VertexFetcherWithMeanCurvatures : public VertexFetcher
{
public:

    struct VertexWithMeanCurvature
    {
        mReal3 r;
        mReal H;
    };
    
    using VertexType = VertexWithMeanCurvature;
    using ViewType   = OVviewWithJuelicherQuants;

    __D__ inline VertexType fetchVertex(const ViewType& view, int i) const
    {
        return {VertexFetcher::fetchVertex(view, i),
                mReal(view.vertexMeanCurvatures[i])};
    }
};
