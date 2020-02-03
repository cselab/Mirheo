#pragma once

#include "exchanger_interfaces.h"

#include <mirheo/core/containers.h>

#include <memory>
#include <vector>
#include <string>

namespace mirheo
{

class ObjectVector;
class ObjectPacker;
class ObjectHaloExchanger;

class ObjectExtraExchanger : public Exchanger
{
public:
    ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    ~ObjectExtraExchanger();

    void attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames);

protected:
    std::vector<ObjectVector*> objects_;
    ObjectHaloExchanger *entangledHaloExchanger_;
    std::vector<std::unique_ptr<ObjectPacker>> packers_, unpackers_;
    
    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
