#pragma once

#include "kernels/common.h"

namespace MembraneForcesKernels
{

struct GPU_CommonMembraneParameters
{
    mReal gammaC, gammaT;
    mReal totArea0, totVolume0;
    mReal ka0, kv0;

    bool fluctuationForces;
    mReal seed, sigma_rnd;
};

__device__ inline mReal3 _fconstrainArea(mReal3 v1, mReal3 v2, mReal3 v3, mReal totArea,
                                         const GPU_CommonMembraneParameters& parameters)
{
    const mReal3 x21 = v2 - v1;
    const mReal3 x32 = v3 - v2;
    const mReal3 x31 = v3 - v1;

    const mReal3 normal = cross(x21, x31);

    const mReal area = 0.5_mr * length(normal);
    const mReal area_1 = 1.0_mr / area;

    const mReal coef = -0.25_mr * parameters.ka0 * (totArea - parameters.totArea0) * area_1;

    return coef * cross(normal, x32);
}

__device__ inline mReal3 _fconstrainVolume(mReal3 v1, mReal3 v2, mReal3 v3, mReal totVolume,
                                          const GPU_CommonMembraneParameters& parameters)
{
    const mReal coeff = parameters.kv0 * (totVolume - parameters.totVolume0);
    return coeff * cross(v3, v2);
}


__device__ inline mReal3 _fvisc(ParticleMReal p1, ParticleMReal p2,
                               const GPU_CommonMembraneParameters& parameters)
{
    const mReal3 du = p2.u - p1.u;
    const mReal3 dr = p1.r - p2.r;

    return du*parameters.gammaT + dr * parameters.gammaC*dot(du, dr) / dot(dr, dr);
}

__device__ inline mReal3 _ffluct(mReal3 v1, mReal3 v2, int i1, int i2,
                                const GPU_CommonMembraneParameters& parameters)
{
    if (!parameters.fluctuationForces)
        return make_mReal3(0.0_mr);

    // mReal mean0var1 = Saru::normal2(parameters.seed, math::min(i1, i2), math::max(i1, i2)).x;

    constexpr mReal sqrt_12 = 3.4641016151_mr;
    const mReal mean0var1 = sqrt_12 * (Saru::uniform01(parameters.seed, math::min(i1, i2), math::max(i1, i2)) - 0.5_mr);

    const mReal3 x21 = v2 - v1;
    return (mean0var1 * parameters.sigma_rnd / length(x21)) * x21;
}

template <class TriangleInteraction>
__device__ inline mReal3 bondTriangleForce(
        const TriangleInteraction& triangleInteraction,
        ParticleMReal p, int locId, int rbcId,
        const OVviewWithAreaVolume& view,
        const MembraneMeshView& mesh,
        const GPU_CommonMembraneParameters& parameters)
{
    mReal3 f0 = make_mReal3(0.0_mr);
    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    const int idv0 = rbcId * mesh.nvertices + locId;
    int idv1 = rbcId * mesh.nvertices + mesh.adjacent[startId];
    auto p1 = fetchParticle(view, idv1);

    const mReal totArea   = view.area_volumes[rbcId].x;
    const mReal totVolume = view.area_volumes[rbcId].y;

#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        const int i1 = startId + i;
        const int i2 = startId + ((i+1) % degree);
        
        const int idv2 = rbcId * mesh.nvertices + mesh.adjacent[i2];

        const auto p2 = fetchParticle(view, idv2);

        const auto eq = triangleInteraction.getEquilibriumDesc(mesh, i1, i2);
        
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
__device__ inline mReal3 dihedralForce(int locId, int rbcId,
                                       const typename DihedralInteraction::ViewType& view,
                                       DihedralInteraction& dihedralInteraction,
                                       const MembraneMeshView& mesh)
{
    const int offset = rbcId * mesh.nvertices;

    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    const int idv0 = offset + locId;
    int idv1 = offset + mesh.adjacent[startId];
    int idv2 = offset + mesh.adjacent[startId+1];

    const auto v0 = dihedralInteraction.fetchVertex(view, idv0);
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
    
    mReal3 f0 = make_mReal3(0.0_mr);

    dihedralInteraction.computeCommon(view, rbcId);

#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        mReal3 f1 = make_mReal3(0.0_mr);
        int idv3 = offset + mesh.adjacent[startId + (i+2) % degree];

        auto v3 = dihedralInteraction.fetchVertex(view, idv3);

        f0 += dihedralInteraction(v0, v1, v2, v3, f1);

        atomicAdd(view.forces + idv1, make_real3(f1));
            
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

    mReal3 f;
    f  = bondTriangleForce(triangleInteraction, p, locId, rbcId, view, mesh, parameters);
    f += dihedralForce(locId, rbcId, dihedralView, dihedralInteraction, mesh);

    atomicAdd(view.forces + pid, make_real3(f));
}

} // namespace MembraneInteractionKernels
