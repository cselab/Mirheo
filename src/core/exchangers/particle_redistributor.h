#pragma once

#include "exchanger_interfaces.h"

#include <core/pvs/extra_data/packers.h>

class ParticleVector;
class CellList;
class ParticlesPacker;

class ParticleRedistributor : public Exchanger
{
public:
    ParticleRedistributor();
    ~ParticleRedistributor();
    
    void attach(ParticleVector *pv, CellList *cl);

private:
    std::vector<ParticleVector*> particles;
    std::vector<CellList*> cellLists;
    std::vector<PackPredicate> packPredicates;
    std::vector<std::unique_ptr<ParticlesPacker>> packers;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};
