#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <core/interactions/utils/step_random_gen.h>
#include <core/utils/cuda_common.h>
#include <core/utils/restart_helpers.h>

#include <random>

class CellList;
class LocalParticleVector;

class PairwiseDPDHandler : public ParticleFetcherWithVelocity
{
public:

    using ViewType     = PVview;
    using ParticleType = Particle;
    
    PairwiseDPDHandler(float rc, float a, float gamma, float sigma, float power) :
        ParticleFetcherWithVelocity(rc),
        a(a),
        gamma(gamma),
        sigma(sigma),
        power(power),
        invrc(1.0 / rc)
    {}
    
    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        const float3 dr = dst.r - src.r;
        const float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        const float invrij = rsqrtf(rij2);
        const float rij = rij2 * invrij;
        const float argwr = 1.0f - rij * invrc;
        const float wr = fastPower(argwr, power);

        const float3 dr_r = dr * invrij;
        const float3 du = dst.u - src.u;
        const float rdotv = dot(dr_r, du);

        const float myrandnr = Logistic::mean0var1(seed, min(src.i1, dst.i1), max(src.i1, dst.i1));

        const float strength = a * argwr - (gamma * wr * rdotv + sigma * myrandnr) * wr;

        return dr_r * strength;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    float a, gamma, sigma, power;
    float invrc;
    float seed;
};

class PairwiseDPD : public PairwiseKernel, public PairwiseDPDHandler
{
public:

    using HandlerType = PairwiseDPDHandler;
    
    PairwiseDPD(float rc, float a, float gamma, float kBT, float dt, float power, long seed=42424242) :
        PairwiseDPDHandler(rc, a, gamma, computeSigma(gamma, kBT, dt), power),
        stepGen(seed),
        kBT(kBT)
    {}

    PairwiseDPD(float rc, const DPDParams& p, float dt, long seed=42424242) :
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

    static float computeSigma(float gamma, float kBT, float dt)
    {
        return sqrt(2.0 * gamma * kBT / dt);
    }
    
    StepRandomGen stepGen;
    float kBT;
};
