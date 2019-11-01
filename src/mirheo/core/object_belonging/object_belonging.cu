#include "object_belonging.h"

#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/packers/particles.h>

#include <mirheo/core/celllist.h>

namespace ObjectBelongingKernels
{

__global__ void countInOut(int n, const BelongingTags *tags, int *nIn, int *nOut)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const BelongingTags tag = tags[i];

    if (tag == BelongingTags::Outside)
        atomicAggInc(nOut);

    if (tag == BelongingTags::Inside)
        atomicAggInc(nIn);
}

__global__ void packToInOut(int n, const BelongingTags *tags, ParticlePackerHandler packer,
                            char *outputInsideBuffer, char *outputOutsideBuffer, int *nIn, int *nOut,
                            int maxInside, int maxOutside)
{
    const int srcPid = blockIdx.x * blockDim.x + threadIdx.x;
    if (srcPid >= n) return;

    const BelongingTags tag = tags[srcPid];

    if (tag == BelongingTags::Outside)
    {
        const int dstPid = atomicAggInc(nOut);
        packer.particles.pack(srcPid, dstPid, outputOutsideBuffer, maxOutside);
    }

    if (tag == BelongingTags::Inside)
    {
        const int dstPid = atomicAggInc(nIn);
        packer.particles.pack(srcPid, dstPid, outputInsideBuffer, maxInside);
    }
}

__global__ void unpackParticles(int n, int dstOffset, const char *inputBuffer, ParticlePackerHandler packer, int maxParticles)
{
    const int srcPid = blockIdx.x * blockDim.x + threadIdx.x;
    if (srcPid >= n) return;
    const int dstPid = dstOffset + srcPid;
    packer.particles.unpack(srcPid, dstPid, inputBuffer, maxParticles);
}

} // namespace ObjectBelongingKernels

ObjectBelongingChecker_Common::ObjectBelongingChecker_Common(const MirState *state, std::string name) :
    ObjectBelongingChecker(state, name)
{}

ObjectBelongingChecker_Common::~ObjectBelongingChecker_Common() = default;


static bool keepAllpersistentDataPredicate(const DataManager::NamedChannelDesc& namedDesc)
{
    return namedDesc.second->persistence == DataManager::PersistenceMode::Active;
};

static void copyToLpv(int start, int n, const char *buffer, LocalParticleVector *lpv, cudaStream_t stream)
{
    if (n <= 0) return;

    constexpr int nthreads = 128;

    ParticlePacker packer(keepAllpersistentDataPredicate);
    packer.update(lpv, stream);
    
    SAFE_KERNEL_LAUNCH(
         ObjectBelongingKernels::unpackParticles,
         getNblocks(n, nthreads), nthreads, 0, stream,
         n, start, buffer, packer.handler(), n);
}

void ObjectBelongingChecker_Common::splitByBelonging(ParticleVector *src, ParticleVector *pvIn, ParticleVector *pvOut, cudaStream_t stream)
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
        PrimaryCellList cl(src, 1.0_r, state->domain.localSize);
        cl.build(stream);
        checkInner(src, &cl, stream);
    }

    info("Splitting PV %s with respect to OV %s. Number of particles: in/out/total %d / %d / %d",
         src->name.c_str(), ov->name.c_str(), nInside[0], nOutside[0], src->local()->size());

    ParticlePacker packer(keepAllpersistentDataPredicate);
    packer.update(src->local(), stream);

    DeviceBuffer<char> insideBuffer (packer.getSizeBytes(nInside [0]));
    DeviceBuffer<char> outsideBuffer(packer.getSizeBytes(nOutside[0]));
    
    nInside. clearDevice(stream);
    nOutside.clearDevice(stream);

    const int srcSize = src->local()->size();
    constexpr int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        ObjectBelongingKernels::packToInOut,
        getNblocks(srcSize, nthreads), nthreads, 0, stream,
        srcSize, tags.devPtr(), packer.handler(), insideBuffer.devPtr(), outsideBuffer.devPtr(),
        nInside.devPtr(), nOutside.devPtr(), nInside[0], nOutside[0] );

    CUDA_Check( cudaStreamSynchronize(stream) );

    if (pvIn  != nullptr)
    {
        const int oldSize = (src == pvIn) ? 0 : pvIn->local()->size();
        pvIn->local()->resize(oldSize + nInside[0], stream);

        copyToLpv(oldSize, nInside[0], insideBuffer.devPtr(), pvIn->local(), stream);

        info("New size of inner PV %s is %d", pvIn->name.c_str(), pvIn->local()->size());
        pvIn->cellListStamp++;
    }

    if (pvOut != nullptr)
    {
        const int oldSize = (src == pvOut) ? 0 : pvOut->local()->size();
        pvOut->local()->resize(oldSize + nOutside[0], stream);

        copyToLpv(oldSize, nOutside[0], outsideBuffer.devPtr(), pvOut->local(), stream);

        info("New size of outer PV %s is %d", pvOut->name.c_str(), pvOut->local()->size());
        pvOut->cellListStamp++;
    }
}

void ObjectBelongingChecker_Common::checkInner(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    tagInner(pv, cl, stream);

    nInside .clear(stream);
    nOutside.clear(stream);

    // Only count
    const int np = pv->local()->size();
    constexpr int nthreads = 128;
    
    SAFE_KERNEL_LAUNCH(
        ObjectBelongingKernels::countInOut,
        getNblocks(np, nthreads), nthreads, 0, stream,
        np, tags.devPtr(), nInside.devPtr(), nOutside.devPtr() );

    nInside. downloadFromDevice(stream, ContainersSynch::Asynch);
    nOutside.downloadFromDevice(stream, ContainersSynch::Synch);

    info("PV %s belonging check against OV %s: in/out/total  %d / %d / %d",
         pv->name.c_str(), ov->name.c_str(), nInside[0], nOutside[0], pv->local()->size());
}

void ObjectBelongingChecker_Common::setup(ObjectVector *ov)
{
    this->ov = ov;
}

std::vector<std::string> ObjectBelongingChecker_Common::getChannelsToBeExchanged() const
{
    return {ChannelNames::motions};
}

ObjectVector* ObjectBelongingChecker_Common::getObjectVector()
{
    return ov;
}
