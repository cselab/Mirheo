#pragma once

#include "exchanger_interfaces.h"
#include <core/containers.h>

class ObjectVector;

class ObjectRedistributor : public ParticleExchanger
{
public:
    void attach(ObjectVector* ov, float rc);

    virtual ~ObjectRedistributor() = default;
    
private:
    std::vector<ObjectVector*> objects;
    
    DeviceBuffer<char> temp;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};
