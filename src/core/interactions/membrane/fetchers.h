#pragma once

#include "real.h"
#include <core/utils/cpu_gpu_defines.h>

class VertexFetcher
{
public:
    using VertexType = real3;
    using ViewType   = OVview;

    __D__ inline VertexType fetchVertex(const ViewType& view, int i) const
    {
        return make_real3(Float3_int(view.readPosition(i)).v);
    }
};

class VertexFetcherWithMeanCurvatures : public VertexFetcher
{
public:

    struct VertexWithMeanCurvature
    {
        real3 r;
        real H;
    };
    
    using VertexType = VertexWithMeanCurvature;
    using ViewType   = OVviewWithJuelicherQuants;

    __D__ inline VertexType fetchVertex(const ViewType& view, int i) const
    {
        return {make_real3(Float3_int(view.particles[2 * i]).v),
                real(view.vertexMeanCurvatures[i])};
    }
};
