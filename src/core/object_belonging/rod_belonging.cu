#include "rod_belonging.h"

#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/rv.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/helper_math.h>

namespace RodBelongingKernels
{

__device__ inline float squaredDistanceToSegment(const float3& r0, const float3& r1, const float3& x)
{
    float3 dr = r1 - r0;
    float alpha = dot(x - r0, dr) / dot(dr, dr);
    alpha = min(1.f, max(0.f, alpha));
    float3 p = r0 + alpha * dr;
    float3 dx = x - p;
    return dot(dx, dx);
}

__device__ inline
void setTagsCell(int pstart, int pend, float radius, const float3& r0, const float3& r1,
                 const PVview& pvView, BelongingTags *tags)
{
    #pragma unroll 2
    for (int pid = pstart; pid < pend; ++pid)
    {
        auto x = make_float3(pvView.readPosition(pid));

        auto d = squaredDistanceToSegment(r0, r1, x);

        bool isInside = d < radius * radius;
        if (!isInside) continue;

        atomicExch((int*) tags + pid, (int) BelongingTags::Inside);
    }
}

__global__ void setInsideTags(RVview rvView, float radius, PVview pvView, CellListInfo cinfo, BelongingTags *tags)
{
    // About maximum distance a particle can cover in one step
    const float tol = 0.25f;

    // One thread per segment
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int rodId = gid / rvView.nSegments;
    const int segId = gid % rvView.nSegments;
    if (rodId >= rvView.nObjects) return;

    int segStart = rodId * rvView.objSize + 5 * segId;
    auto r0 = make_float3(rvView.readPosition(segStart + 0));
    auto r1 = make_float3(rvView.readPosition(segStart + 5));

    const float3 lo = fminf(r0, r1);
    const float3 hi = fmaxf(r0, r1);

    const int3 cidLow  = cinfo.getCellIdAlongAxes(lo - (radius + tol));
    const int3 cidHigh = cinfo.getCellIdAlongAxes(hi + (radius + tol));

    int3 cid3;

    #pragma unroll 2
    for (cid3.z = cidLow.z; cid3.z <= cidHigh.z; ++cid3.z)
    {
        for (cid3.y = cidLow.y; cid3.y <= cidHigh.y; ++cid3.y)
        {
            cid3.x = cidLow.x;
            int cidLo = max(cinfo.encode(cid3), 0);
            
            cid3.x = cidHigh.x;
            int cidHi = min(cinfo.encode(cid3)+1, cinfo.totcells);
            
            int pstart = cinfo.cellStarts[cidLo];
            int pend   = cinfo.cellStarts[cidHi];
            
            setTagsCell(pstart, pend, radius, r0, r1, pvView, tags);
        }
    }
}


} // namespace RodBelongingKernels


RodBelongingChecker::RodBelongingChecker(const MirState *state, std::string name, float radius) :
    ObjectBelongingChecker_Common(state, name),
    radius(radius)
{}

void RodBelongingChecker::tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    auto rv = dynamic_cast<RodVector*> (ov);
    if (rv == nullptr)
        die("Rod belonging can only be used with rod objects (%s is not)", ov->name.c_str());

    tags.resize_anew(pv->local()->size());
    tags.clearDevice(stream);

    const int numSegmentsPerRod = rv->local()->getNumSegmentsPerRod();
    
    auto pvView   = cl->getView<PVview>();

    auto computeTags = [&](ParticleVectorLocality locality)
    {
        auto rvView = RVview(rv, rv->get(locality));

        debug("Computing inside/outside tags for %d %s rods '%s' and %d '%s' particles",
              rvView.nObjects, getParticleVectorLocalityStr(locality).c_str(),
              ov->name.c_str(), pv->local()->size(), pv->name.c_str());

        const int totNumSegments = rvView.nObjects * numSegmentsPerRod;
        constexpr int nthreads = 128;
        const int nblocks = getNblocks(totNumSegments, nthreads);

        SAFE_KERNEL_LAUNCH(
            RodBelongingKernels::setInsideTags,
            nblocks, nthreads, 0, stream,
            rvView, radius, pvView, cl->cellInfo(), tags.devPtr());
    };

    computeTags(ParticleVectorLocality::Local);
    computeTags(ParticleVectorLocality::Halo);
}



