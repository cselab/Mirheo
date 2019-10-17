#include "shape_belonging.h"

#include <core/analytical_shapes/api.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ashape_object_vector.h>
#include <core/pvs/views/rsov.h>
#include <core/rigid/utils.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/quaternion.h>

namespace ShapeBelongingKernels
{

template <class Shape>
__global__ void computeTags(RSOVview<Shape> rsView, CellListInfo cinfo, PVview pvView, BelongingTags *tags)
{
    const float tolerance = 5e-6f;

    const int objId = blockIdx.x;
    const int tid = threadIdx.x;
    if (objId >= rsView.nObjects) return;

    const int3 cidLow  = cinfo.getCellIdAlongAxes<CellListsProjection::Clamp>(rsView.comAndExtents[objId].low  - 1.5f);
    const int3 cidHigh = cinfo.getCellIdAlongAxes<CellListsProjection::Clamp>(rsView.comAndExtents[objId].high + 2.5f);

    const int3 span = cidHigh - cidLow + make_int3(1,1,1);
    const int totCells = span.x * span.y * span.z;

    for (int i = tid; i < totCells; i += blockDim.x)
    {
        const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
        const int  cid = cinfo.encode(cid3);
        if (cid >= cinfo.totcells) continue;

        int pstart = cinfo.cellStarts[cid];
        int pend   = cinfo.cellStarts[cid+1];

        for (int pid = pstart; pid < pend; pid++)
        {
            const Particle p(pvView.readParticle(pid));
            auto motion = toSingleMotion(rsView.motions[objId]);

            float3 coo = Quaternion::rotate(p.r - motion.r, Quaternion::conjugate(motion.q));

            float v = rsView.shape.inOutFunction(coo);

//            if (fabs(v) <= tolerance)
//                tags[pid] = BelongingTags::Boundary;
            if (v <= tolerance)
                tags[pid] = BelongingTags::Inside;
        }
    }
}

} // namespace ShapeBelongingKernels

template <class Shape>
void ShapeBelongingChecker<Shape>::tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    const int nthreads = 512;

    auto rsov = dynamic_cast<RigidShapedObjectVector<Shape>*> (ov);
    if (rsov == nullptr)
        die("%s belonging can only be used with %s objects (%s is not)", Shape::desc, Shape::desc, ov->name.c_str());

    tags.resize_anew(pv->local()->size());
    tags.clearDevice(stream);

    ov->findExtentAndCOM(stream, ParticleVectorLocality::Local);
    ov->findExtentAndCOM(stream, ParticleVectorLocality::Halo);

    auto pvView   = cl->getView<PVview>();
    auto rsovView = RSOVview<Shape>(rsov, rsov->local());

    debug("Computing inside/outside tags for %d local %s '%s' and %d '%s' particles",
          rsovView.nObjects, Shape::desc, ov->name.c_str(), pv->local()->size(), pv->name.c_str());

    SAFE_KERNEL_LAUNCH(
            ShapeBelongingKernels::computeTags,
            rsovView.nObjects, nthreads, 0, stream,
            rsovView, cl->cellInfo(), pvView, tags.devPtr());

    rsovView = RSOVview<Shape>(rsov, rsov->halo());
    debug("Computing inside/outside tags for %d halo %s '%s' and %d '%s' particles",
          rsovView.nObjects, Shape::desc, ov->name.c_str(), pv->local()->size(), pv->name.c_str());

    SAFE_KERNEL_LAUNCH(
            ShapeBelongingKernels::computeTags,
            rsovView.nObjects, nthreads, 0, stream,
            rsovView, cl->cellInfo(), pvView, tags.devPtr());
}

#define INSTANTIATE(Shape) template class ShapeBelongingChecker<Shape>;

ASHAPE_TABLE(INSTANTIATE)
