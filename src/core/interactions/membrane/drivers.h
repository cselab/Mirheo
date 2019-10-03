#pragma once

#include "kernels/common.h"

namespace MembraneForcesKernels
{

struct GPU_CommonMembraneParameters
{
    real gammaC, gammaT;
    real totArea0, totVolume0;
    real ka0, kv0;

    bool fluctuationForces;
    real seed, sigma_rnd;
};

__device__ inline real3 _fconstrainArea(real3 v1, real3 v2, real3 v3, real totArea,
                                        const GPU_CommonMembraneParameters& parameters)
{
    real3 x21 = v2 - v1;
    real3 x32 = v3 - v2;
    real3 x31 = v3 - v1;

    real3 normal = cross(x21, x31);

    real area = 0.5_r * length(normal);
    real area_1 = 1.0_r / area;

    real coef = -0.25_r * parameters.ka0 * (totArea - parameters.totArea0) * area_1;

    return coef * cross(normal, x32);
}

__device__ inline real3 _fconstrainVolume(real3 v1, real3 v2, real3 v3, real totVolume,
                                          const GPU_CommonMembraneParameters& parameters)
{
    real coeff = parameters.kv0 * (totVolume - parameters.totVolume0);
    return coeff * cross(v3, v2);
}


__device__ inline real3 _fvisc(ParticleReal p1, ParticleReal p2,
                               const GPU_CommonMembraneParameters& parameters)
{
    const real3 du = p2.u - p1.u;
    const real3 dr = p1.r - p2.r;

    return du*parameters.gammaT + dr * parameters.gammaC*dot(du, dr) / dot(dr, dr);
}

__device__ inline real3 _ffluct(real3 v1, real3 v2, int i1, int i2,
                                const GPU_CommonMembraneParameters& parameters)
{
    if (!parameters.fluctuationForces)
        return make_real3(0.0_r);

    // real mean0var1 = Saru::normal2(parameters.seed, min(i1, i2), max(i1, i2)).x;

    constexpr real sqrt_12 = 3.4641016151_r;
    real mean0var1 = sqrt_12 * (Saru::uniform01(parameters.seed, min(i1, i2), max(i1, i2)) - 0.5_r);

    real3 x21 = v2 - v1;
    return (mean0var1 * parameters.sigma_rnd / length(x21)) * x21;
}

template <class TriangleInteraction>
__device__ inline real3 bondTriangleForce(
        const TriangleInteraction& triangleInteraction,
        ParticleReal p, int locId, int rbcId,
        const OVviewWithAreaVolume& view,
        const MembraneMeshView& mesh,
        const GPU_CommonMembraneParameters& parameters)
{
    real3 f0 = make_real3(0.0_r);
    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    int idv0 = rbcId * mesh.nvertices + locId;
    int idv1 = rbcId * mesh.nvertices + mesh.adjacent[startId];
    auto p1 = fetchParticle(view, idv1);

    real totArea   = view.area_volumes[rbcId].x;
    real totVolume = view.area_volumes[rbcId].y;

#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        int i1 = startId + i;
        int i2 = startId + ((i+1) % degree);
        
        int idv2 = rbcId * mesh.nvertices + mesh.adjacent[i2];

        auto p2 = fetchParticle(view, idv2);

        auto eq = triangleInteraction.getEquilibriumDesc(mesh, i1, i2);
        
        f0 += triangleInteraction (p.r, p1.r, p2.r, eq)
            + _fconstrainArea     (p.r, p1.r, p2.r, totArea,   parameters)
            + _fconstrainVolume   (p.r, p1.r, p2.r, totVolume, parameters)
            + _fvisc              (p,   p1,                    parameters)
            + _ffluct             (p.r, p1.r, idv0, idv1,      parameters);

        idv1 = idv2;
        p1   = p2;
    }

    return f0;
}

template <class DihedralInteraction>
__device__ inline real3 dihedralForce(int locId, int rbcId,
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

    /*
           v3
         /   \
       v2 --> v0
         \   /
           V
           v1
    */
    
    real3 f0 = make_real3(0.0_r);

    dihedralInteraction.computeCommon(view, rbcId);

#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        real3 f1 = make_real3(0.0_r);
        int idv3 = offset + mesh.adjacent[startId + (i+2) % degree];

        auto v3 = dihedralInteraction.fetchVertex(view, idv3);

        f0 += dihedralInteraction(v0, v1, v2, v3, f1);

        atomicAdd(view.forces + idv1, make_float3(f1));
            
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
                                      GPU_CommonMembraneParameters parameters)
{
    // RBC particles are at the same time mesh vertices
    assert(view.objSize == mesh.nvertices);

    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const int locId = pid % mesh.nvertices;
    const int rbcId = pid / mesh.nvertices;

    if (pid >= view.nObjects * mesh.nvertices) return;

    auto p = fetchParticle(view, pid);

    real3 f;
    f  = bondTriangleForce(triangleInteraction, p, locId, rbcId, view, mesh, parameters);
    f += dihedralForce(locId, rbcId, dihedralView, dihedralInteraction, mesh);

    atomicAdd(view.forces + pid, make_float3(f));
}

} // namespace MembraneInteractionKernels
