#pragma once

#include "exchanger_interfaces.h"

class ObjectVector;
class ObjectHaloExchanger;

class ObjectForcesReverseExchanger : public ParticleExchanger
{
protected:
    std::vector<ObjectVector*> objects;
    ObjectHaloExchanger *entangledHaloExchanger;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

public:
    ObjectForcesReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    virtual ~ObjectForcesReverseExchanger();
    
    void attach(ObjectVector *ov);
};
