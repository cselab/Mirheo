#include "object_reverse_exchanger.h"
#include "exchange_helpers.h"
#include "object_halo_exchanger.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

namespace ObjectReverseExchangerKernels
{
__global__ void pack(int objSize, ParticleExtraPacker packer, char *dstBuffer)
{
    int objId = blockIdx.x;
    int tid = threadIdx.x;

    auto byteSizePerObject = objSize * packer.packedSize_byte;
    
    char *dstAddr = dstBuffer + byteSizePerObject * objId;

    for (int pid = tid; pid < objSize; pid += blockDim.x)
    {
        int srcId = objId * objSize + pid;
        packer.pack(srcId, dstAddr + pid * packer.packedSize_byte);
    }
}

__global__ void unpackAndAdd(int objSize, const int *origins, const char *srcBuffer, ParticleExtraPacker packer)
{
    int objId = blockIdx.x;
    int tid = threadIdx.x;

    auto byteSizePerObject = objSize * packer.packedSize_byte;
    
    const char *srcAddr = srcBuffer + byteSizePerObject * objId;

    for (int pid = tid; pid < objSize; pid += blockDim.x)
    {
        int dstId = origins[objId * objSize + pid];
        packer.unpackAdd(srcAddr + pid * packer.packedSize_byte, dstId);
    }
}
} // namespace ObjectReverseExchangerKernels

ObjectReverseExchanger::ObjectReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger(entangledHaloExchanger)
{}

ObjectReverseExchanger::~ObjectReverseExchanger() = default;

void ObjectReverseExchanger::attach(ObjectVector *ov, const std::vector<std::string>& channelNames)
{
    int id = objects.size();
    objects.push_back(ov);
    int size = 0;

    const auto& extraData = ov->local()->extraPerParticle;

    for (const auto& name : channelNames)
    {
        const auto& desc = extraData.getChannelDescOrDie(name);
        switch (desc.dataType)
        {
        case DataType::TOKENIZE(float):
        case DataType::TOKENIZE(float4):
        case DataType::TOKENIZE(Stress):
            break;
        default:
            die("cannot reverse send data '%s' of type '%s' from ov '%s': should be float data",
                name.c_str(), dataTypeToString(desc.dataType).c_str(), ov->name.c_str());
            break;
        }
    }

    auto helper = std::make_unique<ExchangeHelper>(ov->name, id);
    helper->setDatumSize(size);
        
    helpers.push_back(std::move(helper));

    packPredicates.push_back([channelNames](const ExtraDataManager::NamedChannelDesc& namedDesc) {
        bool isRequired = std::find(channelNames.begin(), channelNames.end(), namedDesc.first) != channelNames.end();
        return isRequired;
    });

}

bool ObjectReverseExchanger::needExchange(int id)
{
    return true;
}

void ObjectReverseExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto  helper  = helpers[id].get();
    auto& offsets = entangledHaloExchanger->getRecvOffsets(id);

    for (int i = 0; i < helper->nBuffers; i++)
        helper->sendSizes[i] = offsets[i+1] - offsets[i];
}

void ObjectReverseExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    int objSize = ov->objSize;
    
    debug2("Preparing '%s' data to sending back", ov->name.c_str());

    ParticleExtraPacker packer(ov, ov->halo(), packPredicates[id], stream);

    helper->setDatumSize(objSize * packer.packedSize_byte);
    helper->computeSendOffsets();
    helper->resizeSendBuf();

    const int nthreads = 128;
    int nObjects = ov->halo()->nObjects;
        
    SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::pack,
            nObjects, nthreads, 0, stream,
            objSize, packer, helper->sendBuf.devPtr() );
    
    debug2("Will send back data for %d objects", helper->sendOffsets[helper->nBuffers]);
}

void ObjectReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    int objSize = ov->objSize;
    
    int totalRecvd = helper->recvOffsets[helper->nBuffers];
    auto& origins = entangledHaloExchanger->getOrigins(id);

    debug("Updating data for %d '%s' objects", totalRecvd, ov->name.c_str());

    ParticleExtraPacker packer(ov, ov->local(), packPredicates[id], stream);
    
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::unpackAndAdd,
            totalRecvd, nthreads, 0, stream,
            objSize,
            (const int*)origins.devPtr(), /* destination ids here */
            helper->recvBuf.devPtr(),     /* source */
            packer);
    
}





