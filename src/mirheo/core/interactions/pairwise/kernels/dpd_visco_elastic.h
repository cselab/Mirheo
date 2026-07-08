// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/forcePolChain.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/interactions/utils/step_random_gen.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/restart_helpers.h>

#include <cmath>
#include <random>

namespace mirheo
{

class CellList;
class LocalParticleVector;

/// a GPU compatible functor that computes DPD interactions
class PairwiseViscoElasticDPDHandler : public ParticleFetcherWithPolChainVectors
{
public:

    using ViewType     = PVviewWithPolChainVector;   ///< compatible view type
    using ParticleType = ParticleWithPolChainVector; ///< compatible particle type

    /// constructor
    PairwiseViscoElasticDPDHandler(real rc, real a, real gamma, real power, real H, real n0) :
        ParticleFetcherWithPolChainVectors(rc),
        a_(a),
        gamma_(gamma),
        power_(power),
        invrc_(1.0_r / rc)
    {
        beta_ = 15.0_r / (rc*rc*rc*rc * static_cast<real>(M_PI) * n0);
        alphaH_ = beta_ * H;
    }

    /// evaluate the force
    __D__ inline ForceDerPolChain operator()(const ParticleType dst, __UNUSED int dstId,
                                             const ParticleType src, __UNUSED int srcId) const
    {
        const real3 dr = dst.p.r - src.p.r;
        const real rij2 = dot(dr, dr);
        if (rij2 > rc2_ || rij2 < 1e-6_r)
            return {make_real3(0.0_r),
                    make_real3(0.0_r),
                    make_real3(0.0_r)};

        const real invrij = math::rsqrt(rij2);
        const real rij = rij2 * invrij;
        const real w  = 1.0_r - rij * invrc_;
        const real wr = fastPower(w, power_);

        const real3 dr_r = dr * invrij;
        const real3 du = dst.p.u - src.p.u;
        const real rdotv = dot(dr_r, du);

        const real myrandnr = Logistic::mean0var1(seed_,
                                                  math::min(static_cast<int>(src.p.i1), static_cast<int>(dst.p.i1)),
                                                  math::max(static_cast<int>(src.p.i1), static_cast<int>(dst.p.i1)));

        const real strength = a_ * w - (gamma_ * wr * rdotv + sigma_ * myrandnr) * wr;

        ForceDerPolChain f;

        f.force = dr_r * strength;
        f.force -= alphaH_ * w * (dot(src.Q, dr_r) * src.Q +
                                  dot(dst.Q, dr_r) * dst.Q);

        f.dQsrc_dt = beta_ * w * dot(src.Q, dr_r) * du;
        f.dQdst_dt = beta_ * w * dot(dst.Q, dr_r) * du;

        return f;
    }


    /// initialize accumulator
    __D__ inline ForceDerPolChainAccumulator getZeroedAccumulator() const {return ForceDerPolChainAccumulator();}


protected:
    real a_; ///< conservative force magnitude
    real gamma_; ///< viscous force coefficient
    real sigma_{NAN}; ///< random force coefficient, depends on dt
    real power_; ///< viscous kernel envelope power
    real alphaH_; ///< Elastic modulus times normalization coefficient
    real beta_; ///< normalization coefficient
    real invrc_; ///< 1 / rc
    real seed_ {0}; ///< random seed, must be updated at every time step
};

/// Helper class that constructs PairwiseViscoElasticDPDHandler
class PairwiseViscoElasticDPD : public PairwiseKernel, public PairwiseViscoElasticDPDHandler
{
public:

    using HandlerType = PairwiseViscoElasticDPDHandler; ///< handler type corresponding to this object
    using ParamsType = ViscoElasticDPDParams; ///< parameters that are used to create this object

    /// Constructor
    PairwiseViscoElasticDPD(real rc, real a, real gamma, real kBT, real power, real H, real n0, long seed=42424242) :
        PairwiseViscoElasticDPDHandler(rc, a, gamma, power, H, n0),
        stepGen_(seed),
        kBT_(kBT)
    {}

    /// Generic constructor
    PairwiseViscoElasticDPD(real rc, const ParamsType& p, long seed=42424242) :
        PairwiseViscoElasticDPD(rc, p.a, p.gamma, p.kBT, p.power, p.H, p.n0, seed)
    {}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&)(*this);
    }

    void setup(__UNUSED LocalParticleVector *lpv1,
               __UNUSED LocalParticleVector *lpv2,
               __UNUSED CellList *cl1,
               __UNUSED CellList *cl2,
               const MirState *state) override
    {
        seed_ = stepGen_.generate(state);
        sigma_ = computeSigma(gamma_, kBT_, state->getDt());
    }

    void writeState(std::ofstream& fout) override
    {
        text_IO::writeToStream(fout, stepGen_);
    }

    bool readState(std::ifstream& fin) override
    {
        return text_IO::readFromStream(fin, stepGen_);
    }

private:
    static real computeSigma(real gamma, real kBT, real dt)
    {
        return math::sqrt(2.0_r * gamma * kBT / dt);
    }

    StepRandomGen stepGen_;
    real kBT_;
};

} // namespace mirheo
