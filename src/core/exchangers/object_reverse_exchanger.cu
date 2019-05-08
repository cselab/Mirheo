#include "object_reverse_exchanger.h"
#include "exchange_helpers.h"
#include "object_halo_exchanger.h"
#include "packers/objects.h"

#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/logger.h>


ObjectReverseExchanger::ObjectReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger(entangledHaloExchanger)
{}

ObjectReverseExchanger::~ObjectReverseExchanger() = default;

void ObjectReverseExchanger::attach(ObjectVector *ov, std::vector<std::string> channelNames)
{
    int id = objects.size();
    objects.push_back(ov);
    
    auto packer = std::make_unique<ObjectsPacker>(ov, [channelNames](const DataManager::NamedChannelDesc& namedDesc) {
        return std::find(channelNames.begin(), channelNames.end(), namedDesc.first) != channelNames.end();
    });
    auto helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());
    
    helpers.push_back(std::move(helper));
    packers.push_back(std::move(packer));
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
        helper->send.sizes[i] = offsets[i+1] - offsets[i];
}

void ObjectReverseExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    
    debug2("Preparing '%s' data to reverse send data");

    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf();

    packer->reversePackToBuffer(ov->halo(), &helper->send, stream);
    
    debug2("Will send back data for %d objects", helper->send.offsets[helper->nBuffers]);
}

void ObjectReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    
    int totalRecvd = helper->recv.offsets[helper->nBuffers];
    auto& map = entangledHaloExchanger->getMap(id);
    
    debug("Updating data for %d '%s' objects", totalRecvd, ov->name.c_str());

    packer->reverseUnpackFromBufferAndAdd(ov->local(), map, &helper->recv, stream);
}
