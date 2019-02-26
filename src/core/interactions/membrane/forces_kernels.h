#pragma once

#include "common.h"

namespace MembraneForcesKernels
{

struct GPU_RBCparameters
{
    float gammaC, gammaT;
    float totArea0, totVolume0;
    float ka0, kv0;

    bool fluctuationForces;
    float seed, sigma_rnd;
};

__device__ inline float3 _fconstrainArea(float3 v1, float3 v2, float3 v3,
                                         float totArea, GPU_RBCparameters parameters)
{
    float3 x21 = v2 - v1;
    float3 x32 = v3 - v2;
    float3 x31 = v3 - v1;

    float3 normal = cross(x21, x31);

    float area = 0.5f * length(normal);
    float area_1 = 1.0f / area;

    float coef = -0.25f * parameters.ka0 * (totArea - parameters.totArea0) * area_1;

    return coef * cross(normal, x32);;
}

__device__ inline float3 _fconstrainVolume(float3 v1, float3 v2, float3 v3, float totVolume, GPU_RBCparameters parameters)
{
    float coeff = parameters.kv0 * (totVolume - parameters.totVolume0);
    return coeff * cross(v3, v2);
}


__device__ inline float3 _fvisc(Particle p1, Particle p2, GPU_RBCparameters parameters)
{
    const float3 du = p2.u - p1.u;
    const float3 dr = p1.r - p2.r;

    return du*parameters.gammaT + dr * parameters.gammaC*dot(du, dr) / dot(dr, dr);
}

__device__ inline float3 _ffluct(float3 v1, float3 v2, int i1, int i2, GPU_RBCparameters parameters)
{
    if (!parameters.fluctuationForces)
        return make_float3(0.0f);

    float2 rnd = Saru::normal2(parameters.seed, min(i1, i2), max(i1, i2));
    float3 x21 = v2 - v1;
    return (rnd.x * parameters.sigma_rnd / length(x21)) * x21;
}

template <class TriangleInteraction>
__device__ inline float3 bondTriangleForce(
        const TriangleInteraction& triangleInteraction,
        Particle p, int locId, int rbcId,
        const OVviewWithAreaVolume& view,
        const MembraneMeshView& mesh,
        const GPU_RBCparameters& parameters)
{
    float3 f = make_float3(0.0f);
    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    int idv0 = rbcId * mesh.nvertices + locId;
    int idv1 = rbcId * mesh.nvertices + mesh.adjacent[startId];
    Particle p1(view.particles, idv1);
    
#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        int i0 = startId + i;
        int i1 = startId + ((i+1) % degree);
        
        int idv2 = rbcId * mesh.nvertices + mesh.adjacent[i1];

        Particle p2(view.particles, idv2);

        auto eq = triangleInteraction.getEquilibriumDesc(mesh, i0, i1);

        float totArea   = view.area_volumes[rbcId].x;
        float totVolume = view.area_volumes[rbcId].y;
        
        f += triangleInteraction (p.r, p1.r, p2.r, eq)
            + _fconstrainArea    (p.r, p1.r, p2.r, totArea,   parameters)
            + _fconstrainVolume  (p.r, p1.r, p2.r, totVolume, parameters)
            + _fvisc     (p,   p1,               parameters)
            + _ffluct    (p.r, p1.r, idv0, idv1, parameters);

        idv1 = idv2;
        p1 = p2;
    }

    return f;
}

template <class DihedralInteraction>
__device__ inline float3 dihedralForce(int locId, int rbcId,
                                       const typename DihedralInteraction::ViewType& view,
                                       DihedralInteraction& dihedralInteraction,
                                       const MembraneMeshView& mesh)
{
    const int offset = rbcId * mesh.nvertices;

    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    int idv0 = offset + locId;
    int idv1 = offset + mesh.adjacent[startId];
    int idv2 = offset + mesh.adjacent[startId+1];

    auto v0 = dihedralInteraction.fetchVertex(view, idv0);
    auto v1 = dihedralInteraction.fetchVertex(view, idv1);
    auto v2 = dihedralInteraction.fetchVertex(view, idv2);

    //       v3
    //     /   \
    //   v2 --> v0
    //     \   /
    //       V
    //       v1

    float3 f0 = make_float3(0,0,0);

    dihedralInteraction.computeCommon(view, rbcId);

#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        float3 f1;
        int idv3 = offset + mesh.adjacent[startId + (i+2) % degree];        

        auto v3 = dihedralInteraction.fetchVertex(view, idv3);

        f0 += dihedralInteraction(v0, v1, v2, v3, f1);

        atomicAdd(view.forces + idv1, f1);
            
        v1   = v2  ; v2   = v3  ;
        idv1 = idv2; idv2 = idv3;
    }
    return f0;
}

template <class TriangleInteraction, class DihedralInteraction>
__global__ void computeMembraneForces(TriangleInteraction triangleInteraction,
                                      DihedralInteraction dihedralInteraction,
                                      typename DihedralInteraction::ViewType dihedralView,
                                      OVviewWithAreaVolume view,
                                      MembraneMeshView mesh,
                                      GPU_RBCparameters parameters)
{
    // RBC particles are at the same time mesh vertices
    assert(view.objSize == mesh.nvertices);

    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const int locId = pid % mesh.nvertices;
    const int rbcId = pid / mesh.nvertices;

    if (pid >= view.nObjects * mesh.nvertices) return;

    Particle p(view.particles, pid);

    float3 f;
    f  = bondTriangleForce(triangleInteraction, p, locId, rbcId, view, mesh, parameters);
    f += dihedralForce(locId, rbcId, dihedralView, dihedralInteraction, mesh);

    atomicAdd(view.forces + pid, f);
}

} // namespace MembraneInteractionKernels
