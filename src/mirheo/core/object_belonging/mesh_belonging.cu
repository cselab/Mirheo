// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "mesh_belonging.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/rigid/utils.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

namespace mesh_belonging_kernels
{

const real tolerance = 1e-6_r;

/// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ static inline bool doesRayIntersectTriangle(
        real3 rayOrigin,
        real3 rayVector,
        real3 v0, real3 v1, real3 v2)
{
    real3 edge1, edge2, h, s, q;
    real a,f,u,v;

    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = cross(rayVector, edge2);
    a = dot(edge1, h);
    if (math::abs(a) < tolerance)
        return false;

    f = 1.0_r / a;
    s = rayOrigin - v0;
    u = f * (dot(s, h));
    if (u < 0.0_r || u > 1.0_r)
        return false;

    q = cross(s, edge1);
    v = f * dot(rayVector, q);
    if (v < 0.0_r || u + v > 1.0_r)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    real t = f * dot(edge2, q);

    if (t > tolerance) // ray intersection
        return true;
    else
        return false; // This means that there is a line intersection but not a ray intersection.
}


__device__ static inline real3 fetchPosition(const real4 *vertices, int i)
{
    auto v = vertices[i];
    return {v.x, v.y, v.z};
}

/**
 * One warp works on one particle
 */
__device__ static inline BelongingTags oneParticleInsideMesh(int pid, real3 r, int objId, const real3 com, const MeshView mesh, const real4* vertices)
{
    // Work in obj reference frame for simplicity
    r = r - com;

    // shoot 3 rays in different directions, count intersections
    constexpr int nRays = 3;
    constexpr real3 rays[nRays] = { {0,1,0}, {0,1,0}, {0,1,0} };
    int counters[nRays] = {0, 0, 0};

    for (int i = laneId(); i < mesh.ntriangles; i += warpSize)
    {
        int3 trid = mesh.triangles[i];

        real3 v0 = fetchPosition(vertices, objId*mesh.nvertices + trid.x) - com;
        real3 v1 = fetchPosition(vertices, objId*mesh.nvertices + trid.y) - com;
        real3 v2 = fetchPosition(vertices, objId*mesh.nvertices + trid.z) - com;

        for (int c = 0; c < nRays; c++)
            if (doesRayIntersectTriangle(r, rays[c], v0, v1, v2))
                counters[c]++;
    }

    // counter is odd if the particle is inside
    // however, realing-point precision sometimes yields in errors
    // so we choose what the majority(!) of the rays say
    int intersecting = 0;
    for (int c = 0; c < nRays; c++)
    {
        counters[c] = warpReduce(counters[c], [] (int a, int b) { return a+b; });
        if ( (counters[c] % 2) != 0 )
            intersecting++;
    }

    if (intersecting > (nRays/2))
        return BelongingTags::Inside;
    else
        return BelongingTags::Outside;
}

/**
 * OVview view is only used to provide # of objects and extent information
 * Actual data is in \p vertices
 * @param cinfo is the cell-list sync'd with the target ParticleVector data
 */
template<int WARPS_PER_OBJ>
__global__ void insideMesh(const OVview ovView, const MeshView mesh, const real4 *vertices, CellListInfo cinfo, PVview pvView, BelongingTags* tags)
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int wid = gid / warpSize;
    const int objId = wid / WARPS_PER_OBJ;

    const int locWid = wid % WARPS_PER_OBJ;

    if (objId >= ovView.nObjects) return;

    const int3 cidLow  = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].low  - 0.5_r);
    const int3 cidHigh = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].high + 0.5_r);

    const int3 span = cidHigh - cidLow + make_int3(1,1,1);
    const int totCells = span.x * span.y * span.z;

    for (int i = locWid; i < totCells; i += WARPS_PER_OBJ)
    {
        const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
        const int  cid = cinfo.encode(cid3);
        if (cid < 0 || cid >= cinfo.totcells) continue;

        int pstart = cinfo.cellStarts[cid];
        int pend   = cinfo.cellStarts[cid+1];

#pragma unroll 3
        for (int pid = pstart; pid < pend; pid++)
        {
            const Particle p(pvView.readParticle(pid));

            auto tag = oneParticleInsideMesh(pid, p.r, objId, ovView.comAndExtents[objId].com, mesh, vertices);

            // Only tag particles inside, default is outside anyways
            if (laneId() == 0 && tag != BelongingTags::Outside)
                tags[pid] = tag;
        }
    }
}

} // namespace mesh_belonging_kernels

void MeshBelongingChecker::_tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    tags_.resize_anew(pv->local()->size());
    tags_.clearDevice(stream);

    auto computeTags = [&](ParticleVectorLocality locality)
    {
        ov_->findExtentAndCOM(stream, locality);

        auto lov = ov_->get(locality);
        auto view = OVview(ov_, lov);
        auto vertices = lov->getMeshVertices(stream);
        auto meshView = MeshView(ov_->mesh.get());

        debug("Computing inside/outside tags (against mesh) for %d %s objects '%s' and %d '%s' particles",
              view.nObjects, getParticleVectorLocalityStr(locality).c_str(),
              ov_->getCName(), pv->local()->size(), pv->getCName());

        constexpr int nthreads = 128;
        constexpr int warpsPerObject = 1024;

        SAFE_KERNEL_LAUNCH(
            mesh_belonging_kernels::insideMesh<warpsPerObject>,
            getNblocks(warpsPerObject*32*view.nObjects, nthreads), nthreads, 0, stream,
            view, meshView, reinterpret_cast<real4*>(vertices->devPtr()),
            cl->cellInfo(), cl->getView<PVview>(), tags_.devPtr());
    };


    computeTags(ParticleVectorLocality::Local);
    computeTags(ParticleVectorLocality::Halo);
}

} // namespace mirheo
