#pragma once

#include "exchanger_interfaces.h"

class ParticleVector;
class CellList;

class ParticleHaloExchanger : public ParticleExchanger
{
private:
    std::vector<CellList*> cellLists;
    std::vector<ParticleVector*> particles;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

public:

    ~ParticleHaloExchanger();
    
    void attach(ParticleVector* pv, CellList* cl);    
};
