#pragma once

#include "particle_exchanger.h"

#include <vector>

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
    ParticleHaloExchanger(MPI_Comm& comm, bool gpuAwareMPI) : ParticleExchanger(comm, gpuAwareMPI) {};

    void attach(ParticleVector* pv, CellList* cl);

    ~ParticleHaloExchanger() = default;
};
