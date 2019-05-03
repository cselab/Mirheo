#include "objects.h"

#include <core/pvs/object_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <type_traits>

namespace ObjectPackerKernels
{

// TODO

} // namespace ObjectPackerKernels

ObjectPacker::ObjectPacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate) :
    Packer(pv, lpv, predicate),    
    ov(dynamic_cast<ObjectVector*>(pv)),
    lov(dynamic_cast<LocalObjectVector*>(lpv))
{
    if (ov == nullptr)
        die("object packers must work with object vectors");

    if (lov == nullptr)
        die("object packers must work with local object vectors");
}

size_t ObjectPacker::getPackedSizeBytes(int nobj)
{
    auto packedSizeParts = _getPackedSizeBytes(lov->dataPerParticle, nobj * ov->objSize);
    auto packedSizeObjs = _getPackedSizeBytes(lov->dataPerObject, nobj);

    return packedSizeParts + packedSizeObjs;
}

void ObjectPacker::packToBuffer(const DeviceBuffer<MapEntry>& map, const PinnedBuffer<int>& sizes,
                                PinnedBuffer<size_t>& offsetsBytes, char *buffer, cudaStream_t stream)
{
    
}

void ObjectPacker::unpackFromBuffer(PinnedBuffer<size_t>& offsetsBytes,
                                    const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                                    const char *buffer, cudaStream_t stream)
{
    
}
