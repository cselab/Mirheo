#pragma once

#include "generic_packer.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <vector>

namespace mirheo
{

class LocalParticleVector;

struct ParticlePackerHandler
{
    GenericPackerHandler particles;

    inline __D__ size_t getSizeBytes(int numElements) const
    {
        return particles.getSizeBytes(numElements);
    }
};

class ParticlePacker
{
public:
    ParticlePacker(PackPredicate predicate);
    ~ParticlePacker();
    
    virtual void update(LocalParticleVector *lpv, cudaStream_t stream);
    ParticlePackerHandler handler();
    virtual size_t getSizeBytes(int numElements) const;

protected:
    PackPredicate predicate_;
    GenericPacker particleData_;
};

} // namespace mirheo
