#pragma once

#include "exchanger_interfaces.h"

namespace mirheo
{

class ParticleVector;
class CellList;
class ParticlePacker;

class ParticleRedistributor : public Exchanger
{
public:
    ParticleRedistributor();
    ~ParticleRedistributor();
    
    void attach(ParticleVector *pv, CellList *cl);

private:
    std::vector<ParticleVector*> particles;
    std::vector<CellList*> cellLists;
    std::vector<std::unique_ptr<ParticlePacker>> packers;

    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
