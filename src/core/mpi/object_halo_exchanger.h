#pragma once

#include <memory>

#include <core/containers.h>
#include <core/pvs/extra_data/packers.h>

#include "exchanger_interfaces.h"


class ObjectVector;

class ObjectHaloExchanger : public ParticleExchanger
{
protected:
    std::vector<float> rcs;
    std::vector<ObjectVector*> objects;
    std::vector<PackPredicate> packPredicates;

    std::vector<std::unique_ptr<PinnedBuffer<int>>> origins;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

public:
    void attach(ObjectVector* ov, float rc);

    PinnedBuffer<int>& getRecvOffsets(int id);
    PinnedBuffer<int>& getOrigins    (int id);

    virtual ~ObjectHaloExchanger();
};
