#pragma once

#include "exchanger_interfaces.h"

#include <mirheo/core/containers.h>

#include <vector>
#include <string>

namespace mirheo
{

class ObjectVector;
class ObjectHaloExchanger;
class ObjectPacker;

class ObjectReverseExchanger : public Exchanger
{
public:
    ObjectReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    virtual ~ObjectReverseExchanger();
    
    void attach(ObjectVector *ov, std::vector<std::string> channelNames);

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
