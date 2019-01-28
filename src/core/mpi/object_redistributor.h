#pragma once

#include "exchanger_interfaces.h"

#include <core/containers.h>
#include <core/pvs/extra_data/packers.h>

class ObjectVector;

class ObjectRedistributor : public ParticleExchanger
{
public:
    void attach(ObjectVector* ov);

    virtual ~ObjectRedistributor() = default;
    
private:
    std::vector<ObjectVector*> objects;
    std::vector<PackPredicate> packPredicates;
    
    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};
