#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include "core/ymero_object.h"

class ParticleVector;
class ObjectVector;
class CellList;

class ObjectBelongingChecker : public YmrSimulationObject
{
public:
    ObjectBelongingChecker(std::string name, const YmrState *state) :
        YmrSimulationObject(name, state)
    {}

    virtual void splitByBelonging(ParticleVector* src, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream) = 0;
    virtual void checkInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) = 0;
    virtual void setup(ObjectVector* ov) = 0;
};
