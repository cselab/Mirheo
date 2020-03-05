#include "rod_belonging.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/views/rv.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace rod_belonging_kernels
{

__device__ inline real squaredDistanceToSegment(const real3& r0, const real3& r1, const real3& x)
{
    real3 dr = r1 - r0;
    real alpha = dot(x - r0, dr) / dot(dr, dr);
    alpha = math::min(1._r, math::max(0._r, alpha));
    real3 p = r0 + alpha * dr;
    real3 dx = x - p;
    return dot(dx, dx);
}

__device__ inline
void setTagsCell(int pstart, int pend, real radius, const real3& r0, const real3& r1,
                 const PVview& pvView, BelongingTags *tags)
{
    #pragma unroll 2
    for (int pid = pstart; pid < pend; ++pid)
    {
        auto x = make_real3(pvView.readPosition(pid));

        auto d = squaredDistanceToSegment(r0, r1, x);

        bool isInside = d < radius * radius;
        if (!isInside) continue;

        atomicExch((int*) tags + pid, (int) BelongingTags::Inside);
    }
}

__global__ void setInsideTags(RVview rvView, real radius, PVview pvView, CellListInfo cinfo, BelongingTags *tags)
{
    // About maximum distance a particle can cover in one step
    const real tol = 0.25_r;

    // One thread per segment
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int rodId = gid / rvView.nSegments;
    const int segId = gid % rvView.nSegments;
    if (rodId >= rvView.nObjects) return;

    int segStart = rodId * rvView.objSize + 5 * segId;
    auto r0 = make_real3(rvView.readPosition(segStart + 0));
    auto r1 = make_real3(rvView.readPosition(segStart + 5));

    const real3 lo = math::min(r0, r1);
    const real3 hi = math::max(r0, r1);

    const int3 cidLow  = cinfo.getCellIdAlongAxes(lo - (radius + tol));
    const int3 cidHigh = cinfo.getCellIdAlongAxes(hi + (radius + tol));

    int3 cid3;

    #pragma unroll 2
    for (cid3.z = cidLow.z; cid3.z <= cidHigh.z; ++cid3.z)
    {
        for (cid3.y = cidLow.y; cid3.y <= cidHigh.y; ++cid3.y)
        {
            cid3.x = cidLow.x;
            const int cidLo = math::max(cinfo.encode(cid3), 0);
            
            cid3.x = cidHigh.x;
            const int cidHi = math::min(cinfo.encode(cid3)+1, cinfo.totcells);
            
            const int pstart = cinfo.cellStarts[cidLo];
            const int pend   = cinfo.cellStarts[cidHi];
            
            setTagsCell(pstart, pend, radius, r0, r1, pvView, tags);
        }
    }
}


} // namespace rod_belonging_kernels


RodBelongingChecker::RodBelongingChecker(const MirState *state, const std::string& name, real radius) :
    ObjectVectorBelongingChecker(state, name),
    radius_(radius)
{}

void RodBelongingChecker::_tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    auto rv = dynamic_cast<RodVector*> (ov_);
    if (rv == nullptr)
        die("Rod belonging can only be used with rod objects (%s is not)", ov_->getCName());

    tags_.resize_anew(pv->local()->size());
    tags_.clearDevice(stream);

    const int numSegmentsPerRod = rv->local()->getNumSegmentsPerRod();
    
    auto pvView   = cl->getView<PVview>();

    auto computeTags = [&](ParticleVectorLocality locality)
    {
        auto rvView = RVview(rv, rv->get(locality));

        debug("Computing inside/outside tags for %d %s rods '%s' and %d '%s' particles",
              rvView.nObjects, getParticleVectorLocalityStr(locality).c_str(),
              ov_->getCName(), pv->local()->size(), pv->getCName());

        const int totNumSegments = rvView.nObjects * numSegmentsPerRod;
        constexpr int nthreads = 128;
        const int nblocks = getNblocks(totNumSegments, nthreads);

        SAFE_KERNEL_LAUNCH(
            rod_belonging_kernels::setInsideTags,
            nblocks, nthreads, 0, stream,
            rvView, radius_, pvView, cl->cellInfo(), tags_.devPtr());
    };

    computeTags(ParticleVectorLocality::Local);
    computeTags(ParticleVectorLocality::Halo);
}

} // namespace mirheo
