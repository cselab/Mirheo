#include "object_halo_extra_exchanger.h"
#include "object_halo_exchanger.h"
#include "exchange_helpers.h"

#include <core/logger.h>
#include <core/pvs/extra_data/packers.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace ObjectHaloExtraExchangeKernels
{
__global__ void pack(const OVview view, const ParticleExtraPacker packer,
                     const int *origins, char *to)
{
    int objId  = blockIdx.x;
    int offset = objId * view.objSize;
    char *dstAddr = to + offset * packer.packedSize_byte;
    
    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
    {
        int srcId = origins[objId * view.objSize + pid];
        packer.pack(srcId, dstAddr + pid * packer.packedSize_byte);
    }
}

__global__ void unpack(const char *from, OVview view, ParticleExtraPacker packer)
{
    int objId = blockIdx.x;
    
    const char* srcAddr = from + packer.packedSize_byte * view.objSize * objId;

    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
    {
        int dstId = objId * view.objSize + pid;
        packer.unpack(srcAddr + pid * packer.packedSize_byte, dstId);
    }
}
} // namespace ObjectHaloExtraExchangeKernels



ObjectExtraExchanger::ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger(entangledHaloExchanger)
{}

ObjectExtraExchanger::~ObjectExtraExchanger() = default;


bool ObjectExtraExchanger::needExchange(int id)
{
    return true;
}

void ObjectExtraExchanger::attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames)
{
    int id = objects.size();
    objects.push_back(ov);

    auto helper = std::make_unique<ExchangeHelper>(ov->name, id);
    helpers.push_back(std::move(helper));

    packPredicates.push_back([extraChannelNames](const DataManager::NamedChannelDesc& namedDesc) {
        return std::find(extraChannelNames.begin(), extraChannelNames.end(), namedDesc.first) != extraChannelNames.end();
    });
}

void ObjectExtraExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto helper = helpers[id].get();

    const auto& offsets = entangledHaloExchanger->getSendOffsets(id);

    for (int i = 0; i < helper->nBuffers; ++i)
        helper->sendSizes[i] = offsets[i+1] - offsets[i];
}

void ObjectExtraExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto helper = helpers[id].get();
    auto& origins = entangledHaloExchanger->getOrigins(id);

    ParticleExtraPacker packer(ov, ov->local(), packPredicates[id], stream);
    OVview view(ov, ov->local());

    helper->setDatumSize(packer.packedSize_byte * view.objSize);
    helper->computeSendOffsets();
    helper->resizeSendBuf();

    int nObjects = helper->sendOffsets[helper->nBuffers];

    const int nthreads = 128;
    
    SAFE_KERNEL_LAUNCH(
        ObjectHaloExtraExchangeKernels::pack,
        nObjects, nthreads, 0, stream,
        view, packer, origins.devPtr(), helper->sendBuf.devPtr() );
}

void ObjectExtraExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();

    int nObjects = ov->halo()->nObjects;
    
    OVview view(ov, ov->halo());
    ParticleExtraPacker packer(ov, ov->halo(), packPredicates[id], stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            ObjectHaloExtraExchangeKernels::unpack,
            nObjects, nthreads, 0, stream,
            helper->recvBuf.devPtr(), view, packer );
}
