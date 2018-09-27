#pragma once

#include "exchanger_interfaces.h"

#include <core/containers.h>

class ObjectVector;

class ObjectHaloExchanger : public ParticleExchanger
{
protected:
    std::vector<float> rcs;
    std::vector<ObjectVector*> objects;

    std::vector<PinnedBuffer<int>*> origins;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

public:
    void attach(ObjectVector* ov, float rc);

    PinnedBuffer<int>& getRecvOffsets(int id);
    PinnedBuffer<int>& getOrigins    (int id);

    virtual ~ObjectHaloExchanger() = default;
};
