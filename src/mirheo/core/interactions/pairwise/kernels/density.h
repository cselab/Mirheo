#pragma once

#include "accumulators/density.h"
#include "density_kernels.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

namespace mirheo
{

class CellList;
class LocalParticleVector;

template <typename DensityKernel>
class PairwiseDensity : public PairwiseKernel, public ParticleFetcher
{
public:

    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleFetcher::ParticleType;
    using HandlerType  = PairwiseDensity;
    using ParamsType   = DensityParams;
    
    PairwiseDensity(real rc, DensityKernel densityKernel) :
        ParticleFetcher(rc),
        densityKernel_(densityKernel),
        invrc_(1.0 / rc)
    {}

    PairwiseDensity(real rc, const ParamsType& p, __UNUSED real dt, __UNUSED long seed=42424242) :
        PairwiseDensity{rc,
                        mpark::get<typename DensityKernel::ParamsType>(p.varDensityKernelParams)}
    {}

    __D__ inline real operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = dst.r - src.r;
        const real rij2 = dot(dr, dr);
        if (rij2 > rc2_)
            return 0.0_r;

        const real rij = math::sqrt(rij2);

        return densityKernel_(rij, invrc_);
    }

    __D__ inline DensityAccumulator getZeroedAccumulator() const {return DensityAccumulator();}


    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
protected:

    real invrc_;
    DensityKernel densityKernel_;
};

} // namespace mirheo
