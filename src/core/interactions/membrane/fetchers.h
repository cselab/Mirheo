#pragma once

#include <core/utils/cpu_gpu_defines.h>

class VertexFetcher
{
public:
    using VertexType = float3;
    using ViewType   = OVview;

    __D__ inline VertexType fetchVertex(ViewType view, int i)
    {
        // 2 because of float4
        return Float3_int(view.particles[2 * i]).v;
    }
};

class VertexFetcherWithMeanCurvatures : public VertexFetcher
{
public:

    struct VertexWithMeanCurvature
    {
        float3 r;
        float H;
    };
    
    using VertexType = VertexWithMeanCurvature;
    using ViewType   = OVviewWithJuelicherQuants;

    __D__ inline VertexType fetchVertex(ViewType view, int i)
    {
        return {Float3_int(view.particles[2 * i]).v,
                view.vertexMeanCurvatures[i]};
    }
};
