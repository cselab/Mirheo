#pragma once

#include "exchanger_interfaces.h"

class ParticleVector;
class CellList;

class ParticleRedistributor : public ParticleExchanger
{
private:
    std::vector<ParticleVector*> particles;
    std::vector<CellList*> cellLists;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

public:
    void _prepareData(int id);
    void attach(ParticleVector* pv, CellList* cl);

    ~ParticleRedistributor() = default;
};
