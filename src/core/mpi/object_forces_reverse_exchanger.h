#pragma once

#include "exchanger_interfaces.h"

#include <core/containers.h>

class ObjectVector;
class ObjectHaloExchanger;

class ObjectForcesReverseExchanger : public ParticleExchanger
{
protected:
    std::vector<ObjectVector*> objects;
    ObjectHaloExchanger *entangledHaloExchanger;

    DeviceBuffer<char>   sortBuffer;
    DeviceBuffer<float4> sortedForces;
    DeviceBuffer<int>    sortedOrigins;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

public:
    ObjectForcesReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    virtual ~ObjectForcesReverseExchanger();
    
    void attach(ObjectVector *ov);
};
