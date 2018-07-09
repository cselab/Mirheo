#pragma once

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;
class ObjectHaloExchanger;

class ObjectForcesReverseExchanger : public ParticleExchanger
{
protected:
    std::vector<ObjectVector*> objects;
    ObjectHaloExchanger* entangledHaloExchanger;

    DeviceBuffer<char>   sortBuffer;
    DeviceBuffer<float4> sortedForces;
    DeviceBuffer<int>    sortedOrigins;

    void prepareSizes(int id, cudaStream_t stream) override;
    void prepareData (int id, cudaStream_t stream) override;
    void combineAndUploadData(int id, cudaStream_t stream) override;
    bool needExchange(int id) override;

public:
    ObjectForcesReverseExchanger(MPI_Comm& comm, ObjectHaloExchanger* entangledHaloExchanger, bool gpuAwareMPI) :
        ParticleExchanger(comm, gpuAwareMPI), entangledHaloExchanger(entangledHaloExchanger)
    { }

    void attach(ObjectVector* ov);

    virtual ~ObjectForcesReverseExchanger() = default;
};
