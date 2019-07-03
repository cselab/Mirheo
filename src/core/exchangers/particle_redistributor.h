#pragma once

#include "exchanger_interfaces.h"

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

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};
