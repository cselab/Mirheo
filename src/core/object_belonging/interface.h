#pragma once

#include <string>
#include <cuda_runtime.h>

class ParticleVector;
class ObjectVector;
class CellList;

class ObjectBelongingChecker
{
public:
    std::string name;

    ObjectBelongingChecker(std::string name) : name(name) { }

    virtual void splitByBelonging(ParticleVector* src, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream) = 0;
    virtual void checkInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) = 0;
    virtual void setup(ObjectVector* ov) = 0;
    virtual ~ObjectBelongingChecker() = default;
};
