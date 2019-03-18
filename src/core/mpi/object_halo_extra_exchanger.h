#pragma once

#include "exchanger_interfaces.h"

#include <core/containers.h>
#include <core/pvs/extra_data/packers.h>

#include <vector>
#include <string>

class ObjectVector;
class ObjectHaloExchanger;

class ObjectExtraExchanger : public ParticleExchanger
{
public:
    ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    virtual ~ObjectExtraExchanger();

    void attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames);

protected:
    std::vector<ObjectVector*> objects;
    ObjectHaloExchanger *entangledHaloExchanger;
    std::vector<PackPredicate> packPredicates;    
    
    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};
