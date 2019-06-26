#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"

#include <core/interactions/utils/step_random_gen.h>
#include <core/utils/cuda_common.h>
#include <core/utils/restart_helpers.h>
#include <core/mirheo_state.h>

#include <random>

class CellList;
class LocalParticleVector;


class PairwiseMDPDHandler : public ParticleFetcherWithVelocityAndDensity
{
public:

    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensity;
    
    PairwiseMDPDHandler(float rc, float rd, float a, float b, float gamma, float kbT, float dt, float power) :
        ParticleFetcherWithVelocityAndDensity(rc),
        rd(rd), a(a), b(b), gamma(gamma), power(power)
    {
        sigma = sqrt(2 * gamma * kbT / dt);
        invrc = 1.0 / rc;
        invrd = 1.0 / rd;
    }

    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        float3 dr = dst.p.r - src.p.r;
        float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        float invrij = rsqrtf(rij2);
        float rij = rij2 * invrij;
        float argwr = 1.0f - rij * invrc;
        float argwd = max(1.0f - rij * invrd, 0.f);

        float wr = fastPower(argwr, power);

        float3 dr_r = dr * invrij;
        float3 du = dst.p.u - src.p.u;
        float rdotv = dot(dr_r, du);

        float myrandnr = Logistic::mean0var1(seed, min(src.p.i1, dst.p.i1), max(src.p.i1, dst.p.i1));

        float strength = a * argwr + b * argwd * (src.d + dst.d) - (gamma * wr * rdotv + sigma * myrandnr) * wr;

        return dr_r * strength;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    float a, b, gamma, sigma, power, rd;
    float invrc, invrd;
    float seed;
};

class PairwiseMDPD : public PairwiseKernel, public PairwiseMDPDHandler
{
public:

    using HandlerType = PairwiseMDPDHandler;
    
    PairwiseMDPD(float rc, float rd, float a, float b, float gamma, float kbT, float dt, float power, long seed = 42424242) :
        PairwiseMDPDHandler(rc, rd, a, b, gamma, kbT, dt, power),
        stepGen(seed)
    {}

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const MirState *state) override
    {
        seed = stepGen.generate(state);
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

    StepRandomGen stepGen;
};
