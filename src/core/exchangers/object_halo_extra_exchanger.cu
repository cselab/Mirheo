#include "object_halo_extra_exchanger.h"
#include "object_halo_exchanger.h"
#include "exchange_helpers.h"
#include "packers/objects.h"

#include <core/logger.h>
#include <core/pvs/object_vector.h>

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

    auto packer = std::make_unique<ObjectsPacker>(ov, [extraChannelNames](const DataManager::NamedChannelDesc& namedDesc) {
        return std::find(extraChannelNames.begin(), extraChannelNames.end(), namedDesc.first) != extraChannelNames.end();
    });

    auto helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());

    packers.push_back(std::move(packer));
    helpers.push_back(std::move(helper));
}

void ObjectExtraExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    const auto& offsets = entangledHaloExchanger->getSendOffsets(id);

    for (int i = 0; i < helper->nBuffers; ++i)
        helper->send.sizes[i] = offsets[i+1] - offsets[i];
}

void ObjectExtraExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov     = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    auto& map   = entangledHaloExchanger->getMap(id);

    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf();

    packer->packToBuffer(ov->local(), map, &helper->send, stream);
}

void ObjectExtraExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    packer->unpackFromBuffer(ov->halo(), &helper->recv, 0, stream);
}
