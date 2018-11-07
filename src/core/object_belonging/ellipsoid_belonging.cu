#include "ellipsoid_belonging.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/views/reov.h>
#include <core/celllist.h>

#include <core/rigid_kernels/quaternion.h>
#include <core/rigid_kernels/rigid_motion.h>


__device__ inline float ellipsoidF(const float3 r, const float3 invAxes)
{
    return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1.0f;
}

__global__ void insideEllipsoid(REOVview view, CellListInfo cinfo, BelongingTags* tags)
{
    const float tolerance = 5e-6f;

    const int objId = blockIdx.x;
    const int tid = threadIdx.x;
    if (objId >= view.nObjects) return;

    const int3 cidLow  = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].low  - 1.5f);
    const int3 cidHigh = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].high + 2.5f);

    const int3 span = cidHigh - cidLow + make_int3(1,1,1);
    const int totCells = span.x * span.y * span.z;

    for (int i=tid; i<totCells; i+=blockDim.x)
    {
        const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
        const int  cid = cinfo.encode(cid3);
        if (cid >= cinfo.totcells) continue;

        int pstart = cinfo.cellStarts[cid];
        int pend   = cinfo.cellStarts[cid+1];

        for (int pid = pstart; pid < pend; pid++)
        {
            const Particle p(cinfo.particles, pid);
            auto motion = toSingleMotion(view.motions[objId]);

            float3 coo = rotate(p.r - motion.r, invQ(motion.q));

            float v = ellipsoidF(coo, view.invAxes);

//            if (fabs(v) <= tolerance)
//                tags[pid] = BelongingTags::Boundary;
            if (v <= tolerance)
                tags[pid] = BelongingTags::Inside;
        }
    }
}


void EllipsoidBelongingChecker::tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream)
{
    int nthreads = 512;

    auto reov = dynamic_cast<RigidEllipsoidObjectVector*> (ov);
    if (reov == nullptr)
        die("Ellipsoid belonging can only be used with ellipsoid objects (%s is not)", ov->name.c_str());

    tags.resize_anew(pv->local()->size());
    tags.clearDevice(stream);

    ov->findExtentAndCOM(stream, ParticleVectorType::Local);
    ov->findExtentAndCOM(stream, ParticleVectorType::Halo);

    auto view = REOVview(reov, reov->local());
    debug("Computing inside/outside tags for %d local ellipsoids '%s' and %d '%s' particles",
          view.nObjects, ov->name.c_str(), pv->local()->size(), pv->name.c_str());

    SAFE_KERNEL_LAUNCH(
            insideEllipsoid,
            view.nObjects, nthreads, 0, stream,
            view, cl->cellInfo(), tags.devPtr());

    view = REOVview(reov, reov->halo());
    debug("Computing inside/outside tags for %d halo ellipsoids '%s' and %d '%s' particles",
          view.nObjects, ov->name.c_str(), pv->local()->size(), pv->name.c_str());

    SAFE_KERNEL_LAUNCH(
            insideEllipsoid,
            view.nObjects, nthreads, 0, stream,
            view, cl->cellInfo(), tags.devPtr());
}



