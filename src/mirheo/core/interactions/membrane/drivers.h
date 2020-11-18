// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "force_kernels/common.h"

namespace mirheo
{

namespace membrane_forces_kernels
{

/// Device compatible structure that holds the parameters common to all membrane interactions
struct GPUConstraintMembraneParameters
{
    mReal totArea0;   ///< total area at equilibrium
    mReal totVolume0; ///< total volume at equilibrium
    mReal ka0; ///< energy magnitude for total area constraint
    mReal kv0; ///< energy magnitude for total volume constraint
};

struct GPUViscMembraneParameters
{
    mReal gammaC; ///< viscous coefficient, central
    mReal gammaT; ///< viscous coefficient, tangential

    mReal seed;      ///< seed that is used for rng; must be changed at every time interation
    mReal sigma_rnd; ///< random force coefficient
};

__device__ inline mReal3 _fconstrainArea(mReal3 v1, mReal3 v2, mReal3 v3, mReal totArea,
                                         const GPUConstraintMembraneParameters& parameters)
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
                                          const GPUConstraintMembraneParameters& parameters)
{
    const mReal coeff = parameters.kv0 * (totVolume - parameters.totVolume0);
    return coeff * cross(v3, v2);
}


template <class TriangleInteraction>
__device__ inline mReal3 triangleForce(
        const TriangleInteraction& triangleInteraction,
        const mReal3& r0, int locId, int rbcId,
        const OVviewWithAreaVolume& view,
        const MembraneMeshView& mesh,
        const GPUConstraintMembraneParameters& parameters)
{
    mReal3 f0 = make_mReal3(0.0_mr);
    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    const int idv1 = rbcId * mesh.nvertices + mesh.adjacent[startId];
    auto r1 = fetchPosition(view, idv1);

    const mReal totArea   = view.area_volumes[rbcId].x;
    const mReal totVolume = view.area_volumes[rbcId].y;

#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        const int i1 = startId + i;
        const int i2 = startId + ((i+1) % degree);

        const int idv2 = rbcId * mesh.nvertices + mesh.adjacent[i2];

        const auto r2 = fetchPosition(view, idv2);

        const auto eq = triangleInteraction.getEquilibriumDesc(mesh, i1, i2);

        f0 += triangleInteraction (r0, r1, r2, eq)
            + _fconstrainArea     (r0, r1, r2, totArea,   parameters)
            + _fconstrainVolume   (r0, r1, r2, totVolume, parameters);

        r1 = r2;
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

    dihedralInteraction.computeInternalCommonQuantities(view, rbcId);

#pragma unroll 2
    for (int i = 0; i < degree; ++i)
    {
        mReal3 f1 = make_mReal3(0.0_mr);
        const int idv3 = offset + mesh.adjacent[startId + (i+2) % degree];

        const auto v3 = dihedralInteraction.fetchVertex(view, idv3);

        f0 += dihedralInteraction(v0, v1, v2, v3, f1);

        atomicAdd(view.forces + idv1, make_real3(f1));

        v1   = v2  ; v2   = v3  ;
        idv1 = idv2; idv2 = idv3;
    }
    return f0;
}

template <class TriangleInteraction, class DihedralInteraction, class Filter>
__global__ void computeMembraneForces(TriangleInteraction triangleInteraction,
                                      DihedralInteraction dihedralInteraction,
                                      typename DihedralInteraction::ViewType dihedralView,
                                      OVviewWithAreaVolume view,
                                      MembraneMeshView mesh,
                                      GPUConstraintMembraneParameters parameters,
                                      Filter filter)
{
    // RBC particles are at the same time mesh vertices
    assert(view.objSize == mesh.nvertices);

    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const int locId = pid % mesh.nvertices;
    const int rbcId = pid / mesh.nvertices;

    if (pid >= view.nObjects * mesh.nvertices) return;
    if (!filter.inWhiteList(rbcId)) return;

    const auto r0 = fetchPosition(view, pid);

    mReal3 f;
    f  = triangleForce(triangleInteraction, r0, locId, rbcId, view, mesh, parameters);
    f += dihedralForce(locId, rbcId, dihedralView, dihedralInteraction, mesh);

    atomicAdd(view.forces + pid, make_real3(f));
}




__device__ inline mReal3 _fvisc(ParticleMReal p1, ParticleMReal p2,
                               const GPUViscMembraneParameters& parameters)
{
    const mReal3 du = p2.u - p1.u;
    const mReal3 dr = p1.r - p2.r;

    return du*parameters.gammaT + dr * parameters.gammaC*dot(du, dr) / dot(dr, dr);
}

__device__ inline mReal3 _ffluct(mReal3 v1, mReal3 v2, int i1, int i2,
                                 const GPUViscMembraneParameters& parameters)
{
    constexpr mReal sqrt_12 = 3.4641016151_mr;
    const mReal mean0var1 = sqrt_12 * (Saru::uniform01(parameters.seed, math::min(i1, i2), math::max(i1, i2)) - 0.5_mr);

    const mReal3 x21 = v2 - v1;
    return (mean0var1 * parameters.sigma_rnd / length(x21)) * x21;
}

__device__ inline mReal3 bondForces(
        const ParticleMReal& p, int locId, int rbcId,
        const OVview& view,
        const MembraneMeshView& mesh,
        const GPUViscMembraneParameters& parameters)
{
    mReal3 f0 = make_mReal3(0.0_mr);
    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    const int idv0 = rbcId * mesh.nvertices + locId;

#pragma unroll 2
    for (int d = 0; d < degree; ++d)
    {
        const int i1 = startId + d;

        const int idv1 = rbcId * mesh.nvertices + mesh.adjacent[i1];

        const auto p1 = fetchParticle(view, idv1);

        f0 +=  _fvisc  (p,   p1,               parameters)
            +  _ffluct (p.r, p1.r, idv0, idv1, parameters);
    }

    return f0;
}

template <class Filter>
__global__ void computeMembraneViscousFluctForces(OVview view,
                                                  MembraneMeshView mesh,
                                                  GPUViscMembraneParameters parameters,
                                                  Filter filter)
{
    assert(view.objSize == mesh.nvertices);
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const int locId = pid % mesh.nvertices;
    const int rbcId = pid / mesh.nvertices;

    if (pid >= view.nObjects * mesh.nvertices) return;
    if (!filter.inWhiteList(rbcId)) return;

    const auto p = fetchParticle(view, pid);

    const mReal3 f = bondForces(p, locId, rbcId, view, mesh, parameters);

    atomicAdd(view.forces + pid, make_real3(f));
}

} // namespace membrane_forces_kernels
} // namespace mirheo
