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
    std::vector<ObjectVector*> objects_;
    std::vector<std::unique_ptr<ObjectPacker>> packers_;
    
    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
