#pragma once

#include "exchanger_interfaces.h"

class ParticleVector;
class CellList;
class ParticlePacker;

class ParticleHaloExchanger : public Exchanger
{
public:
    ParticleHaloExchanger();
    ~ParticleHaloExchanger();
    
    void attach(ParticleVector *pv, CellList *cl, const std::vector<std::string>& extraChannelNames);

private:
    std::vector<CellList*> cellLists;
    std::vector<ParticleVector*> particles;
    std::vector<std::unique_ptr<ParticlePacker>> packers;
    std::vector<std::unique_ptr<ParticlePacker>> unpackers;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;
};
