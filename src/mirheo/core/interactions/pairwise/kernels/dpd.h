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

class PairwiseDPDHandler : public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    
    PairwiseDPDHandler(real rc, real a, real gamma, real sigma, real power) :
        ParticleFetcherWithVelocity(rc),
        a_(a),
        gamma_(gamma),
        sigma_(sigma),
        power_(power),
        invrc_(1.0 / rc)
    {}
    
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

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}


protected:
    real a_, gamma_, sigma_, power_;
    real invrc_;
    real seed_ {0};
};

class PairwiseDPD : public PairwiseKernel, public PairwiseDPDHandler
{
public:

    using HandlerType = PairwiseDPDHandler;
    using ParamsType = DPDParams;
    
    PairwiseDPD(real rc, real a, real gamma, real kBT, real dt, real power, long seed=42424242) :
        PairwiseDPDHandler(rc, a, gamma, computeSigma(gamma, kBT, dt), power),
        stepGen_(seed),
        kBT_(kBT)
    {}

    PairwiseDPD(real rc, const ParamsType& p, real dt, long seed=42424242) :
        PairwiseDPD(rc, p.a, p.gamma, p.kBT, dt, p.power, seed)
    {}

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
        TextIO::writeToStream(fout, stepGen_);
    }

    bool readState(std::ifstream& fin) override
    {
        return TextIO::readFromStream(fin, stepGen_);
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
