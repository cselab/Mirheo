#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include "core/udevicex_object.h"

class ParticleVector;
class ObjectVector;
class CellList;

class ObjectBelongingChecker : public UdxSimulationObject
{
public:
    ObjectBelongingChecker(std::string name) : UdxSimulationObject(name) { }

    virtual void splitByBelonging(ParticleVector* src, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream) = 0;
    virtual void checkInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) = 0;
    virtual void setup(ObjectVector* ov) = 0;
};
