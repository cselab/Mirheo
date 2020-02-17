#include "shape_belonging.h"

#include <mirheo/core/analytical_shapes/api.h>
#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>
#include <mirheo/core/pvs/views/rsov.h>
#include <mirheo/core/rigid/utils.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

namespace ShapeBelongingKernels
{

template <class Shape>
__global__ void computeTags(RSOVview<Shape> rsView, CellListInfo cinfo, PVview pvView, BelongingTags *tags)
{
    const real tolerance = 5e-6_r;

    const int objId = blockIdx.x;
    const int tid = threadIdx.x;
    if (objId >= rsView.nObjects) return;

    const int3 cidLow  = cinfo.getCellIdAlongAxes<CellListsProjection::Clamp>(rsView.comAndExtents[objId].low  - 1.5_r);
    const int3 cidHigh = cinfo.getCellIdAlongAxes<CellListsProjection::Clamp>(rsView.comAndExtents[objId].high + 2.5_r);

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
            const auto motion = toRealMotion(rsView.motions[objId]);

            const real3 coo = motion.q.inverseRotate(p.r - motion.r);

            const real v = rsView.shape.inOutFunction(coo);

            if (v <= tolerance)
                tags[pid] = BelongingTags::Inside;
        }
    }
}

} // namespace ShapeBelongingKernels

template <class Shape>
void ShapeBelongingChecker<Shape>::_tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    auto rsov = dynamic_cast<RigidShapedObjectVector<Shape>*> (ov_);
    if (rsov == nullptr)
        die("%s belonging can only be used with %s objects (%s is not)", Shape::desc, Shape::desc, ov_->getCName());

    tags_.resize_anew(pv->local()->size());
    tags_.clearDevice(stream);

    auto pvView = cl->getView<PVview>();

    auto computeTags = [&](ParticleVectorLocality locality)
    {
        ov_->findExtentAndCOM(stream, locality);
        
        auto rsovView = RSOVview<Shape>(rsov, rsov->get(locality));

        debug("Computing inside/outside tags for %d %s %s '%s' and %d '%s' particles",
              rsovView.nObjects, getParticleVectorLocalityStr(locality).c_str(),
              Shape::desc, ov_->getCName(), pv->local()->size(), pv->getCName());

        constexpr int nthreads = 512;

        SAFE_KERNEL_LAUNCH(
            ShapeBelongingKernels::computeTags,
            rsovView.nObjects, nthreads, 0, stream,
            rsovView, cl->cellInfo(), pvView, tags_.devPtr());
    };        

    computeTags(ParticleVectorLocality::Local);
    computeTags(ParticleVectorLocality::Halo);
}

#define INSTANTIATE(Shape) template class ShapeBelongingChecker<Shape>;
ASHAPE_TABLE(INSTANTIATE)
#undef INSTANTIATE

} // namespace mirheo
