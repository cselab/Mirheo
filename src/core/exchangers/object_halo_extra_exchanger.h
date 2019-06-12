#pragma once

#include "exchanger_interfaces.h"

#include <core/containers.h>

#include <vector>
#include <string>

class ObjectVector;
class ObjectsPacker;
class ObjectHaloExchanger;

class ObjectExtraExchanger : public Exchanger
{
public:
    ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    ~ObjectExtraExchanger();

    void attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames);

protected:
    std::vector<ObjectVector*> objects;
    ObjectHaloExchanger *entangledHaloExchanger;
    std::vector<std::unique_ptr<ObjectsPacker>> packers;
    
    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};
