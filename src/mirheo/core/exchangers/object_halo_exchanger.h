#pragma once

#include "exchanger_interfaces.h"
#include "utils/map.h"

#include <mirheo/core/containers.h>

#include <memory>

namespace mirheo
{

class ObjectVector;
class ObjectPacker;
class MapEntry;

class ObjectHaloExchanger : public Exchanger
{
public:
    ObjectHaloExchanger();
    ~ObjectHaloExchanger();

    void attach(ObjectVector *ov, real rc, const std::vector<std::string>& extraChannelNames);

    PinnedBuffer<int>& getSendOffsets(size_t id);
    PinnedBuffer<int>& getRecvOffsets(size_t id);
    DeviceBuffer<MapEntry>& getMap   (size_t id);

protected:
    std::vector<real> rcs_;
    std::vector<ObjectVector*> objects_;
    std::vector<std::unique_ptr<ObjectPacker>> packers_, unpackers_;
    std::vector<DeviceBuffer<MapEntry>> maps_;

    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
