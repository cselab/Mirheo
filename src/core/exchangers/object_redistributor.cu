#include "object_redistributor.h"
#include "exchange_helpers.h"
#include "packers/map.h"
#include "packers/objects.h"
#include "utils/fragments_mapping.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/pvs/extra_data/packers.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

enum class PackMode
{
    Query, Pack
};

namespace ObjecRedistributorKernels
{

template <PackMode packMode>
__global__ void getExitingObjectsMap(const DomainInfo domain, OVview view, MapEntry *map, BufferOffsetsSizesWrap dataWrap)
{
    const int objId = threadIdx.x + blockIdx.x * blockDim.x;

    if (objId >= view.nObjects) return;

    // Find to which buffer this object should go
    auto prop = view.comAndExtents[objId];
    int dx = 0, dy = 0, dz = 0;

    if (prop.com.x  < -0.5f*domain.localSize.x) dx = -1;
    if (prop.com.y  < -0.5f*domain.localSize.y) dy = -1;
    if (prop.com.z  < -0.5f*domain.localSize.z) dz = -1;

    if (prop.com.x >=  0.5f*domain.localSize.x) dx = 1;
    if (prop.com.y >=  0.5f*domain.localSize.y) dy = 1;
    if (prop.com.z >=  0.5f*domain.localSize.z) dz = 1;

    const int bufId = FragmentMapping::getId(dx, dy, dz);

    int dstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

    if (packMode == PackMode::Query)
    {
        return;
    }
    else
    {
        int dstId = dataWrap.offsets[bufId] + dstObjId;
        map[dstId] = MapEntry(objId, bufId);
    }
}

} // namespace ObjecRedistributorKernels

//===============================================================================================
// Member functions
//===============================================================================================

ObjectRedistributor::ObjectRedistributor() = default;
ObjectRedistributor::~ObjectRedistributor() = default;

bool ObjectRedistributor::needExchange(int id)
{
    return !objects[id]->redistValid;
}

void ObjectRedistributor::attach(ObjectVector *ov)
{
    int id = objects.size();
    objects.push_back(ov);

    auto packer = std::make_unique<ObjectsPacker>(ov, [](const DataManager::NamedChannelDesc& namedDesc) {
        return namedDesc.second->persistence == DataManager::PersistenceMode::Persistent;
    });

    auto helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());

    packers.push_back(std::move(packer));
    helpers.push_back(std::move(helper));

    info("The Object vector '%s' was attached to redistributor", ov->name.c_str());
}


void ObjectRedistributor::prepareSizes(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto bulkId = helper->bulkId;
    
    ov->findExtentAndCOM(stream, ParticleVectorType::Local);
    
    OVview ovView(ov, ov->local());

    debug2("Counting exiting objects of '%s'", ov->name.c_str());
    const int nthreads = 32;

    // Prepare sizes
    helper->send.sizes.clear(stream);
    
    if (ovView.nObjects > 0)
    {
        SAFE_KERNEL_LAUNCH(
            ObjecRedistributorKernels::getExitingObjectsMap<PackMode::Query>,
            getNblocks(ovView.nObjects, nthreads), nthreads, 0, stream,
            ov->state->domain, ovView, nullptr, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }

    int nObjs = helper->send.sizes[bulkId];
    debug2("%d objects of '%s' will leave", ovView.nObjects - nObjs, ov->name.c_str());

    // Early termination support
    if (nObjs == ovView.nObjects)
    {
        helper->send.sizes[bulkId] = 0;
        helper->computeSendOffsets();
        helper->send.uploadInfosToDevice(stream);
        helper->resizeSendBuf();
    }
}

void ObjectRedistributor::prepareData(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto bulkId = helper->bulkId;
    auto packer = packers[id].get();

    OVview ovView(ov, ov->local());

    const int nthreads = 32;
    int nObjsBulk = helper->send.sizes[bulkId];

    // Early termination - no redistribution
    if (helper->send.offsets[helper->nBuffers] == 0)
    {
        debug2("No objects of '%s' leaving, no need to rebuild the object vector", ov->name.c_str());
        return;
    }

    debug2("Downloading %d leaving objects of '%s'", ovView.nObjects - nObjsBulk, ov->name.c_str());

    // Gather data
    helper->resizeSendBuf();
    helper->send.sizes.clearDevice(stream);
    helper->map.resize_anew(ovView.nObjects);
    
    SAFE_KERNEL_LAUNCH(
        ObjecRedistributorKernels::getExitingObjectsMap<PackMode::Pack>,
        getNblocks(ovView.nObjects, nthreads), nthreads, 0, stream,
        ov->state->domain, ovView, helper->map.devPtr(), helper->wrapSendData() );

    packer->packToBuffer(lov, helper->map, &helper->send, stream);
    

    // Unpack the central buffer into the object vector itself
    // Renew view and packer, as the ObjectVector may have resized
    lov->resize_anew(nObjsBulk * ov->objSize);

    packer->unpackFromBuffer(lov, &helper->send, 0, stream);
    
    helper->send.sizes[bulkId] = 0;
    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf(); // relying here on the fact that bulkId is the last one
}

void ObjectRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    int oldNObjs = ov->local()->nObjects;
    int objSize = ov->objSize;

    int totalRecvd = helper->recv.offsets[helper->nBuffers];

    ov->local()->resize((oldNObjs + totalRecvd) * objSize, stream);

    packer->unpackFromBuffer(ov->local(), &helper->recv, oldNObjs, stream);

    ov->redistValid = true;

    // Particles may have migrated, rebuild cell-lists
    if (totalRecvd > 0)
        ov->cellListStamp++;
}



