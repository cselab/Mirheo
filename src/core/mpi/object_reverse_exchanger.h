#pragma once

#include "exchanger_interfaces.h"

#include <core/containers.h>
#include <core/pvs/extra_data/packers.h>

#include <vector>
#include <string>

class ObjectVector;
class ObjectHaloExchanger;

class ObjectReverseExchanger : public ParticleExchanger
{
public:
    ObjectReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    virtual ~ObjectReverseExchanger();
    
    void attach(ObjectVector *ov, std::vector<std::string> channelNames);

protected:
    std::vector<ObjectVector*> objects;    
    ObjectHaloExchanger *entangledHaloExchanger;
    std::vector<PackPredicate> packPredicates;
    std::vector<bool> needForces;
    
    
    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

    int getForceDatumSize(int id) const;
};
