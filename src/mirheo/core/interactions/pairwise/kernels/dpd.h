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
        a(a),
        gamma(gamma),
        sigma(sigma),
        power(power),
        invrc(1.0 / rc)
    {}
    
    __D__ inline real3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const real3 dr = dst.r - src.r;
        const real rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_real3(0.0_r);

        const real invrij = math::rsqrt(rij2);
        const real rij = rij2 * invrij;
        const real argwr = 1.0_r - rij * invrc;
        const real wr = fastPower(argwr, power);

        const real3 dr_r = dr * invrij;
        const real3 du = dst.u - src.u;
        const real rdotv = dot(dr_r, du);

        const real myrandnr = Logistic::mean0var1(seed,
                                                  math::min(static_cast<int>(src.i1), static_cast<int>(dst.i1)),
                                                  math::max(static_cast<int>(src.i1), static_cast<int>(dst.i1)));

        const real strength = a * argwr - (gamma * wr * rdotv + sigma * myrandnr) * wr;

        return dr_r * strength;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    real a, gamma, sigma, power;
    real invrc;
    real seed;
};

class PairwiseDPD : public PairwiseKernel, public PairwiseDPDHandler
{
public:

    using HandlerType = PairwiseDPDHandler;
    
    PairwiseDPD(real rc, real a, real gamma, real kBT, real dt, real power, long seed=42424242) :
        PairwiseDPDHandler(rc, a, gamma, computeSigma(gamma, kBT, dt), power),
        stepGen(seed),
        kBT(kBT)
    {}

    PairwiseDPD(real rc, const DPDParams& p, real dt, long seed=42424242) :
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
        this->seed = stepGen.generate(state);
        sigma = computeSigma(gamma, kBT, state->dt);
    }

    void writeState(std::ofstream& fout) override
    {
        TextIO::writeToStream(fout, stepGen);
    }

    bool readState(std::ifstream& fin) override
    {
        return TextIO::readFromStream(fin, stepGen);
    }
    

protected:

    static real computeSigma(real gamma, real kBT, real dt)
    {
        return math::sqrt(2.0 * gamma * kBT / dt);
    }
    
    StepRandomGen stepGen;
    real kBT;
};

} // namespace mirheo
