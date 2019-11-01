#include "object_deleter.h"

#include <mirheo/core/utils/kernel_launch.h>
#include <extern/cub/cub/device/device_scan.cuh>

namespace ObjectDeleterDetails
{

static bool allChannelsPredicate(const DataManager::NamedChannelDesc &)
{
    return true;
}

__global__ void removeMarkedObjects(ObjectPackerHandler src, ObjectPackerHandler dst, bool *marks, int *prefixSum, int numObjects)
{
    int i = blockIdx.x;
    // if (i >= numObjects) return;  // Not necessary.

    if (marks[i])
        return;
    int srcIdx = i;
    int dstIdx = i - prefixSum[i];
    src.blockCopyTo(dst, srcIdx, dstIdx);
}

/*
__global__ void copyMarkedObjectParticles(ObjectPackerHandler src, ParticlePackerHandler dst, int oldNumParticles, bool *marks, int *prefixSum)
{
    int i = blockIdx.x;
    // if (i >= numObjects) return;  // Not necessary.

    if (!marks[i])
        return;
    int srcObjId = i;
    int dstPartIdOffset = oldNumParticles + (i - prefixSum[i]) * src.objSize;
    src.blockCopyParticlesTo(dst, srcObjId, dstPartIdOffset);
}
*/

} // namespace ObjectDeleterDetails


ObjectDeleter::ObjectDeleter() :
    packerSrc(ObjectDeleterDetails::allChannelsPredicate),
    packerDst(ObjectDeleterDetails::allChannelsPredicate),
    partPackerDst(ObjectDeleterDetails::allChannelsPredicate)
{}

ObjectDeleter::~ObjectDeleter() = default;

void ObjectDeleter::update(LocalObjectVector *lov, cudaStream_t stream)
{
    if (!tmpLov || tmpLov->objSize != lov->objSize)
        tmpLov = std::make_unique<LocalObjectVector>(lov->pv, lov->objSize);

    marks.resize_anew(lov->nObjects + 1);
    marks.clearDevice(stream);
    tmpLov->dataPerObject  .copyChannelMap(lov->dataPerObject);
    tmpLov->dataPerParticle.copyChannelMap(lov->dataPerParticle);
}

ObjectDeleterHandler ObjectDeleter::handler()
{
    return ObjectDeleterHandler{marks.devPtr()};
}


void ObjectDeleter::deleteObjects(LocalObjectVector *lov, cudaStream_t stream, LocalParticleVector *lpvTarget)
{
    // Example:
    //
    //      indices:         0   1   2   3   4   5   6
    //      marks         =  0   0   1   0   0   1   1
    //      prefixSum     =  0   0   1   1   1   2   3
    //      mapping       =  0   1  -1   2   3  -1  -1
    //      mapping[i] = marks[i] ? -1 : i - prefixSum[i];

    const int nthreads = 128;
    int numOld = lov->nObjects;
    if (numOld == 0)
        return;
    prefixSum.resize_anew(numOld + 1);

    size_t bufferSize = scanBuffer.size();
    cub::DeviceScan::ExclusiveSum(nullptr, bufferSize, marks.devPtr(), prefixSum.devPtr(), numOld + 1, stream);
    scanBuffer.resize_anew(bufferSize);
    cub::DeviceScan::ExclusiveSum(scanBuffer.devPtr(), bufferSize, marks.devPtr(), prefixSum.devPtr(), numOld + 1, stream);

    // Copy the total number of removed objects.
    CUDA_Check( cudaMemcpyAsync(numRemoved.hostPtr(), prefixSum.devPtr() + numOld,
                                sizeof(decltype(prefixSum)::value_type), cudaMemcpyDeviceToHost, stream) );
    CUDA_Check( cudaStreamSynchronize(stream) );

    if (numRemoved[0] == 0)
        return;

    int numNew = lov->nObjects - numRemoved[0];

    // Just in case, swap before updating packers and calling the kernel.
    swap(*lov, *tmpLov.get());

    lov->resize_anew(numNew * lov->objSize);
    packerSrc.update(tmpLov.get(), stream);
    packerDst.update(lov, stream);

    // If requested, copy the object particles to another particle vector.
    if (lpvTarget != nullptr) {
        die("lpvTarget not implemented yet.");  // beforeCelllists vs afterIntegration problem.

        /*
        // Not sure about the performance of this part...
        partPackerDst.update(lpvTarget, stream);
        lpvTarget->resize(lpvTarget->size() + numNew * lov->objSize, stream);
        SAFE_KERNEL_LAUNCH(
            ObjectDeleterDetails::copyMarkedObjectParticles,
            numOld, nthreads, 0, stream,
            packerSrc.handler(), partPackerDst.handler(), lpvTarget->size(),
            marks.devPtr(), prefixSum.devPtr());
        */
    }

    SAFE_KERNEL_LAUNCH(
        ObjectDeleterDetails::removeMarkedObjects,
        numOld, nthreads, 0, stream,
        packerSrc.handler(), packerDst.handler(), marks.devPtr(), prefixSum.devPtr(), numOld);
}
