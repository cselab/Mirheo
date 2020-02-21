#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/mirheo_state.h>

#include <random>

namespace mirheo
{

class LocalParticleVector;
class CellList;

/// a GPU compatible functor that computes DPD interactions without fluctuations.
/// Used in unit tests
class PairwiseNorandomDPD : public PairwiseKernel, public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;   ///< compatible view type
    using ParticleType = Particle; ///< compatible particle type
    using HandlerType  = PairwiseNorandomDPD;  ///< handler type corresponding to this object
    using ParamsType   = NoRandomDPDParams; ///< parameters that are used to create this object
    
    /// constructor
    PairwiseNorandomDPD(real rc, real a, real gamma, real kBT, real dt, real power) :
        ParticleFetcherWithVelocity(rc),
        a_(a),
        gamma_(gamma),
        sigma_(math::sqrt(2 * gamma_ * kBT / dt)),
        power_(power),
        invrc_(1.0 / rc)
    {}

    /// Generic constructor
    PairwiseNorandomDPD(real rc, const ParamsType& p, real dt, long seed=42424242) :
        PairwiseNorandomDPD(rc, p.a, p.gamma, p.kBT, dt, p.power)
    {}

    /// evaluate the force
    __D__ inline real3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = dst.r - src.r;
        const real rij2 = dot(dr, dr);
        if (rij2 > rc2_) return make_real3(0.0_r);

        const real invrij = math::rsqrt(rij2);
        const real rij = rij2 * invrij;
        const real argwr = 1.0_r - rij * invrc_;
        const real wr = fastPower(argwr, power_);

        const real3 dr_r = dr * invrij;
        const real3 du = dst.u - src.u;
        const real rdotv = dot(dr_r, du);

        const real myrandnr = ((math::min((int)src.i1, (int)dst.i1) ^ math::max((int)src.i1, (int)dst.i1)) % 13) - 6;

        const real strength = a_ * argwr - (gamma_ * wr * rdotv + sigma_ * myrandnr) * wr;

        return dr_r * strength;
    }

    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

    /// \return type name string
    static std::string getTypeName()
    {
        return "PairwiseNorandomDPD";
    }
    
protected:
    real a_; ///< conservative force magnitude
    real gamma_; ///< viscous force coefficient
    real sigma_; ///< random force coefficient
    real power_; ///< viscous kernel envelope power
    real invrc_; ///< 1 / rc
};


} // namespace mirheo
