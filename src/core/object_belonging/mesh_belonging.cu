#include "mesh_belonging.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/celllist.h>

#include <core/rigid_kernels/quaternion.h>
#include <core/rigid_kernels/rigid_motion.h>

const float tolerance = 1e-6f;

/// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ inline bool doesRayIntersectTriangle(
        float3 rayOrigin,
        float3 rayVector,
        float3 v0, float3 v1, float3 v2)
{
    float3 edge1, edge2, h, s, q;
    float a,f,u,v;

    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = cross(rayVector, edge2);
    a = dot(edge1, h);
    if (fabs(a) < tolerance)
        return false;

    f = 1.0f / a;
    s = rayOrigin - v0;
    u = f * (dot(s, h));
    if (u < 0.0f || u > 1.0f)
        return false;

    q = cross(s, edge1);
    v = f * dot(rayVector, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);

    if (t > tolerance) // ray intersection
        return true;
    else
        return false; // This means that there is a line intersection but not a ray intersection.
}


/**
 * One warp works on one particle
 */
__device__ BelongingTags oneParticleInsideMesh(int pid, float3 r, int objId, const float3 com, const MeshView mesh, const float4* vertices)
{
    // Work in obj reference frame for simplicity
    r = r - com;

    // shoot 3 rays in different directions, count intersections
    const int nRays = 3;
    float3 rays[nRays] = { {0,1,0}, {0,1,0}, {0,1,0} };
    int counters[nRays] = {0, 0, 0};

    for (int i = __laneid(); i < mesh.ntriangles; i += warpSize)
    {
        int3 trid = mesh.triangles[i];

        float3 v0 = Particle(vertices, objId*mesh.nvertices + trid.x).r - com;
        float3 v1 = Particle(vertices, objId*mesh.nvertices + trid.y).r - com;
        float3 v2 = Particle(vertices, objId*mesh.nvertices + trid.z).r - com;

//        if (threadIdx.x == 0 && blockIdx.x == 0)
//            printf("%d  %f %f %f\n", trid, v0.x, v0.y, v0.z);

        for (int c=0; c<nRays; c++)
            if (doesRayIntersectTriangle(r, rays[c], v0, v1, v2)) counters[c]++;
    }

    // counter is odd if the particle is inside
    // however, floating-point precision sometimes yields in errors
    // so we choose what the majority(!) of the rays say
    int intersecting = 0;
    for (int c=0; c<nRays; c++)
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
__global__ void insideMesh(const OVview view, const MeshView mesh, float4* vertices, CellListInfo cinfo, BelongingTags* tags)
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int wid = gid / warpSize;
    const int objId = wid / WARPS_PER_OBJ;

    const int locWid = wid % WARPS_PER_OBJ;

    if (objId >= view.nObjects) return;

    const int3 cidLow  = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].low  - 0.5f);
    const int3 cidHigh = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].high + 0.5f);

    const int3 span = cidHigh - cidLow + make_int3(1,1,1);
    const int totCells = span.x * span.y * span.z;

    for (int i=locWid; i<totCells; i+=WARPS_PER_OBJ)
    {
        const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
        const int  cid = cinfo.encode(cid3);
        if (cid < 0 || cid >= cinfo.totcells) continue;

        int pstart = cinfo.cellStarts[cid];
        int pend   = cinfo.cellStarts[cid+1];

#pragma unroll 3
        for (int pid = pstart; pid < pend; pid++)
        {
            const Particle p(cinfo.particles, pid);

            auto tag = oneParticleInsideMesh(pid, p.r, objId, view.comAndExtents[objId].com, mesh, vertices);

            // Only tag particles inside, default is outside anyways
            if (__laneid() == 0 && tag != BelongingTags::Outside)
                tags[pid] = tag;
        }
    }
}


void MeshBelongingChecker::tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream)
{
    int nthreads = 128;

    tags.resize_anew(pv->local()->size());
    tags.clearDevice(stream);

    const int warpsPerObject = 1024;

    ov->findExtentAndCOM(stream, ParticleVectorType::Local);
    ov->findExtentAndCOM(stream, ParticleVectorType::Halo);

    // Local
    auto lov = ov->local();
    auto view = OVview(ov, lov);
    auto vertices = lov->getMeshVertices(stream);
    auto meshView = MeshView(ov->mesh.get());

    debug("Computing inside/outside tags (against mesh) for %d local objects '%s' and %d '%s' particles",
          view.nObjects, ov->name().c_str(), pv->local()->size(), pv->name().c_str());

    SAFE_KERNEL_LAUNCH(
            insideMesh<warpsPerObject>,
            getNblocks(warpsPerObject*32*view.nObjects, nthreads), nthreads, 0, stream,
            view, meshView, (float4*)vertices->devPtr(), cl->cellInfo(), tags.devPtr());

    // Halo
    lov = ov->halo();       // Note ->halo() here
    view = OVview(ov, lov);
    vertices = lov->getMeshVertices(stream);
    meshView = MeshView(ov->mesh.get());

    debug("Computing inside/outside tags (against mesh) for %d halo objects '%s' and %d '%s' particles",
          view.nObjects, ov->name().c_str(), pv->local()->size(), pv->name().c_str());

    SAFE_KERNEL_LAUNCH(
            insideMesh<warpsPerObject>,
            getNblocks(warpsPerObject*32*view.nObjects, nthreads), nthreads, 0, stream,
            view, meshView, (float4*)vertices->devPtr(), cl->cellInfo(), tags.devPtr());
}



