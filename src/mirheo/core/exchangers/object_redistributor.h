#pragma once

#include "exchanger_interfaces.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class ObjectVector;
class ObjectPacker;

class ObjectRedistributor : public Exchanger
{
public:
    ObjectRedistributor();
    ~ObjectRedistributor();

    void attach(ObjectVector *ov);
    
private:
    std::vector<ObjectVector*> objects;
    std::vector<std::unique_ptr<ObjectPacker>> packers;
    
    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};

} // namespace mirheo
