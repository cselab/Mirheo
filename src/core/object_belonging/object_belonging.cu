#include "object_belonging.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/pvs/object_vector.h>

#include <core/celllist.h>

__global__ void copyInOut(
        PVview view,
        const BelongingTags* tags,
        Particle* ins, Particle* outs,
        int* nIn, int* nOut)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    auto tag = tags[gid];
    const Particle p(view.particles, gid);

    if (tag == BelongingTags::Outside)
    {
        int dstId = atomicAggInc(nOut);
        if (outs) outs[dstId] = p;
    }

    if (tag == BelongingTags::Inside)
    {
        int dstId = atomicAggInc(nIn);
        if (ins)  ins [dstId] = p;
    }
}

void ObjectBelongingChecker_Common::splitByBelonging(ParticleVector* src, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream)
{
    if (dynamic_cast<ObjectVector*>(src) != nullptr)
        error("Trying to split object vector %s into two per-particle, probably that's not what you wanted",
                src->name.c_str());

    if (pvIn != nullptr && typeid(*src) != typeid(*pvIn))
        error("PV type of inner result of split (%s) is different from source (%s)",
                pvIn->name.c_str(), src->name.c_str());

    if (pvOut != nullptr && typeid(*src) != typeid(*pvOut))
        error("PV type of outer result of split (%s) is different from source (%s)",
                pvOut->name.c_str(), src->name.c_str());

    {
        PrimaryCellList cl(src, 1.0f, src->domain.localSize);
        cl.build(stream);
        checkInner(src, &cl, stream);
    }

    info("Splitting PV %s with respect to OV %s. Number of particles: in/out/total %d / %d / %d",
                src->name.c_str(), ov->name.c_str(), nInside[0], nOutside[0], src->local()->size());

    // Need buffers because the source is the same as inside or outside
    PinnedBuffer<Particle> bufIn(nInside[0]), bufOut(nOutside[0]);

    nInside. clearDevice(stream);
    nOutside.clearDevice(stream);
    tags.downloadFromDevice(stream);

    PVview view(src, src->local());
    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            copyInOut,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view,
            tags.devPtr(), bufIn.devPtr(), bufOut.devPtr(),
            nInside.devPtr(), nOutside.devPtr() );

    CUDA_Check( cudaStreamSynchronize(stream) );

    if (pvIn  != nullptr)
    {
        int oldSize = (src == pvIn) ? 0 : pvIn->local()->size();
        pvIn->local()->resize(oldSize + nInside[0], stream);

        if (nInside[0] > 0)
            CUDA_Check( cudaMemcpyAsync(pvIn->local()->coosvels.devPtr() + oldSize,
                    bufIn.devPtr(),
                    nInside[0] * sizeof(Particle),
                    cudaMemcpyDeviceToDevice, stream) );


        info("New size of inner PV %s is %d", pvIn->name.c_str(), pvIn->local()->size());
    }

    if (pvOut != nullptr)
    {
        int oldSize = (src == pvOut) ? 0 : pvOut->local()->size();
        pvOut->local()->resize(oldSize + nOutside[0], stream);

        if (nOutside[0] > 0)
            CUDA_Check( cudaMemcpyAsync(pvOut->local()->coosvels.devPtr() + oldSize,
                    bufOut.devPtr(),
                    nOutside[0] * sizeof(Particle),
                    cudaMemcpyDeviceToDevice, stream) );


        info("New size of outer PV %s is %d", pvOut->name.c_str(), pvOut->local()->size());
    }
}

void ObjectBelongingChecker_Common::checkInner(ParticleVector* pv, CellList* cl, cudaStream_t stream)
{
    tagInner(pv, cl, stream);

    nInside.clear(stream);
    nOutside.clear(stream);

    // Only count
    PVview view(pv, pv->local());
    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
                copyInOut,
                getNblocks(view.size, nthreads), nthreads, 0, stream,
                view, tags.devPtr(), nullptr, nullptr,
                nInside.devPtr(), nOutside.devPtr() );

    nInside. downloadFromDevice(stream, ContainersSynch::Asynch);
    nOutside.downloadFromDevice(stream, ContainersSynch::Synch);

    say("PV %s belonging check against OV %s: in/out/total  %d / %d / %d",
            pv->name.c_str(), ov->name.c_str(), nInside[0], nOutside[0], pv->local()->size());
}
