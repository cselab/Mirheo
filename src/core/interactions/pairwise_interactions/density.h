#pragma once

#include "density_kernels.h"
#include "fetchers.h"

#include <core/interactions/accumulators/density.h>
#include <core/ymero_state.h>

class CellList;
class LocalParticleVector;

template <typename DensityKernel>
class PairwiseDensity : public ParticleFetcher
{
public:

    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleFetcher::ParticleType;
    using HandlerType  = PairwiseDensity;
    
    PairwiseDensity(float rc, DensityKernel densityKernel) :
        ParticleFetcher(rc),
        densityKernel(densityKernel)
    {
        invrc = 1.0 / rc;
    }

    __D__ inline float operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        float3 dr = dst.r - src.r;
        float rij2 = dot(dr, dr);
        if (rij2 > rc2) return 0.0f;

        float rij = sqrtf(rij2);

        return densityKernel(rij, invrc);
    }

    __D__ inline DensityAccumulator getZeroedAccumulator() const {return DensityAccumulator();}


    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const YmrState *state)
    {}

protected:

    float invrc;
    DensityKernel densityKernel;
};
