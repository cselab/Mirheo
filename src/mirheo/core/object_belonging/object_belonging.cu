// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "object_belonging.h"

#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/packers/particles.h>

#include <mirheo/core/celllist.h>

namespace mirheo
{

namespace object_belonging_kernels
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

} // namespace object_belonging_kernels

ObjectVectorBelongingChecker::ObjectVectorBelongingChecker(const MirState *state, const std::string& name) :
    ObjectBelongingChecker(state, name)
{}

ObjectVectorBelongingChecker::~ObjectVectorBelongingChecker() = default;


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
         object_belonging_kernels::unpackParticles,
         getNblocks(n, nthreads), nthreads, 0, stream,
         n, start, buffer, packer.handler(), n);
}

void ObjectVectorBelongingChecker::splitByBelonging(ParticleVector *src, ParticleVector *pvIn, ParticleVector *pvOut, cudaStream_t stream)
{
    if (dynamic_cast<ObjectVector*>(src) != nullptr)
        error("Trying to split object vector %s into two per-particle, probably that's not what you wanted",
              src->getCName());

    if (pvIn != nullptr && typeid(*src) != typeid(*pvIn))
        error("PV type of inner result of split (%s) is different from source (%s)",
              pvIn->getCName(), src->getCName());

    if (pvOut != nullptr && typeid(*src) != typeid(*pvOut))
        error("PV type of outer result of split (%s) is different from source (%s)",
              pvOut->getCName(), src->getCName());

    {
        PrimaryCellList cl(src, 1.0_r, getState()->domain.localSize);
        cl.build(stream);
        checkInner(src, &cl, stream);
    }

    info("Splitting PV %s with respect to OV %s. Number of particles: in/out/total %d / %d / %d",
         src->getCName(), ov_->getCName(), nInside_[0], nOutside_[0], src->local()->size());

    ParticlePacker packer(keepAllpersistentDataPredicate);
    packer.update(src->local(), stream);

    DeviceBuffer<char> insideBuffer (packer.getSizeBytes(nInside_ [0]));
    DeviceBuffer<char> outsideBuffer(packer.getSizeBytes(nOutside_[0]));

    nInside_. clearDevice(stream);
    nOutside_.clearDevice(stream);

    const int srcSize = src->local()->size();
    constexpr int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        object_belonging_kernels::packToInOut,
        getNblocks(srcSize, nthreads), nthreads, 0, stream,
        srcSize, tags_.devPtr(), packer.handler(), insideBuffer.devPtr(), outsideBuffer.devPtr(),
        nInside_.devPtr(), nOutside_.devPtr(), nInside_[0], nOutside_[0] );

    CUDA_Check( cudaStreamSynchronize(stream) );

    if (pvIn  != nullptr)
    {
        const int oldSize = (src == pvIn) ? 0 : pvIn->local()->size();
        pvIn->local()->resize(oldSize + nInside_[0], stream);

        copyToLpv(oldSize, nInside_[0], insideBuffer.devPtr(), pvIn->local(), stream);

        info("New size of inner PV %s is %d", pvIn->getCName(), pvIn->local()->size());
        pvIn->cellListStamp++;
    }

    if (pvOut != nullptr)
    {
        const int oldSize = (src == pvOut) ? 0 : pvOut->local()->size();
        pvOut->local()->resize(oldSize + nOutside_[0], stream);

        copyToLpv(oldSize, nOutside_[0], outsideBuffer.devPtr(), pvOut->local(), stream);

        info("New size of outer PV %s is %d", pvOut->getCName(), pvOut->local()->size());
        pvOut->cellListStamp++;
    }
}

void ObjectVectorBelongingChecker::checkInner(ParticleVector *pv, CellList *cl, cudaStream_t stream)
{
    _tagInner(pv, cl, stream);

    nInside_ .clear(stream);
    nOutside_.clear(stream);

    // Only count
    const int np = pv->local()->size();
    constexpr int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        object_belonging_kernels::countInOut,
        getNblocks(np, nthreads), nthreads, 0, stream,
        np, tags_.devPtr(), nInside_.devPtr(), nOutside_.devPtr() );

    nInside_. downloadFromDevice(stream, ContainersSynch::Asynch);
    nOutside_.downloadFromDevice(stream, ContainersSynch::Synch);

    info("PV %s belonging check against OV %s: in/out/total  %d / %d / %d",
         pv->getCName(), ov_->getCName(), nInside_[0], nOutside_[0], pv->local()->size());
}

void ObjectVectorBelongingChecker::setup(ObjectVector *ov)
{
    ov_ = ov;
}

std::vector<std::string> ObjectVectorBelongingChecker::getChannelsToBeExchanged() const
{
    return {channel_names::motions};
}

ObjectVector* ObjectVectorBelongingChecker::getObjectVector()
{
    return ov_;
}

} // namespace mirheo
