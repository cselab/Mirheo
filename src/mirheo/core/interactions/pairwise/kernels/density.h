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

/** \brief Compute number density from pairwise kernel
    \tparam DensityKernel The kernel used to evaluate the number density
*/
template <typename DensityKernel>
class PairwiseDensity : public PairwiseKernel, public ParticleFetcher
{
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS // breathe warnings
    using ViewType     = PVviewWithDensities;           ///< compatible view type
    using ParticleType = ParticleFetcher::ParticleType; ///< compatible particle type
    using HandlerType  = PairwiseDensity; ///< handler type corresponding to this object
    using ParamsType   = DensityParams;   ///< parameters that are used to create this object
#endif // DOXYGEN_SHOULD_SKIP_THIS
    
    /// construct from density kernel
    PairwiseDensity(real rc, DensityKernel densityKernel) :
        ParticleFetcher(rc),
        densityKernel_(densityKernel),
        invrc_(1.0 / rc)
    {}

    /// generic constructor
    PairwiseDensity(real rc, const ParamsType& p, __UNUSED real dt, __UNUSED long seed=42424242) :
        PairwiseDensity{rc,
                        mpark::get<typename DensityKernel::ParamsType>(p.varDensityKernelParams)}
    {}

    /// evaluate the number density contribution of this pair
    __D__ inline real operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = dst.r - src.r;
        const real rij2 = dot(dr, dr);
        if (rij2 > rc2_)
            return 0.0_r;

        const real rij = math::sqrt(rij2);

        return densityKernel_(rij, invrc_);
    }

    /// initialize the accumulator
    __D__ inline DensityAccumulator getZeroedAccumulator() const {return DensityAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

    /// \return type name string
    static std::string getTypeName()
    {
        return constructTypeName<DensityKernel>("PairwiseDensity");
    }
    
protected:
    real invrc_; ///< 1 / rc
    DensityKernel densityKernel_; ///< the underlying density kernel
};

} // namespace mirheo
