// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/interactions/utils/step_random_gen.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/restart_helpers.h>

#include <random>

namespace mirheo
{

class CellList;
class LocalParticleVector;

/// a GPU compatible functor that computes DPD interactions
class PairwiseDPDHandler : public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;   ///< compatible view type
    using ParticleType = Particle; ///< compatible particle type

    /// constructor
    PairwiseDPDHandler(real rc, real a, real gamma, real sigma, real power) :
        ParticleFetcherWithVelocity(rc),
        a_(a),
        gamma_(gamma),
        sigma_(sigma),
        power_(power),
        invrc_(1.0 / rc)
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

        const real myrandnr = Logistic::mean0var1(seed_,
                                                  math::min(static_cast<int>(src.i1), static_cast<int>(dst.i1)),
                                                  math::max(static_cast<int>(src.i1), static_cast<int>(dst.i1)));

        const real strength = a_ * argwr - (gamma_ * wr * rdotv + sigma_ * myrandnr) * wr;

        return dr_r * strength;
    }


    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}


protected:
    real a_; ///< conservative force magnitude
    real gamma_; ///< viscous force coefficient
    real sigma_; ///< random force coefficient
    real power_; ///< viscous kernel envelope power
    real invrc_; ///< 1 / rc
    real seed_ {0}; ///< random seed, must be updated at every time step
};

/// Helper class that constructs PairwiseDPDHandler
class PairwiseDPD : public PairwiseKernel, public PairwiseDPDHandler
{
public:

    using HandlerType = PairwiseDPDHandler; ///< handler type corresponding to this object
    using ParamsType = DPDParams; ///< parameters that are used to create this object

    /// Constructor
    PairwiseDPD(real rc, real a, real gamma, real kBT, real dt, real power, long seed=42424242) :
        PairwiseDPDHandler(rc, a, gamma, computeSigma(gamma, kBT, dt), power),
        stepGen_(seed),
        kBT_(kBT)
    {}

    /// Generic constructor
    PairwiseDPD(real rc, const ParamsType& p, real dt, long seed=42424242) :
        PairwiseDPD(rc, p.a, p.gamma, p.kBT, dt, p.power, seed)
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
        sigma_ = computeSigma(gamma_, kBT_, state->dt);
    }

    void writeState(std::ofstream& fout) override
    {
        text_IO::writeToStream(fout, stepGen_);
    }

    bool readState(std::ifstream& fin) override
    {
        return text_IO::readFromStream(fin, stepGen_);
    }

    /// \return type name string
    static std::string getTypeName()
    {
        return "PairwiseDPD";
    }

private:
    static real computeSigma(real gamma, real kBT, real dt)
    {
        return math::sqrt(2.0 * gamma * kBT / dt);
    }

    StepRandomGen stepGen_;
    real kBT_;
};

} // namespace mirheo
