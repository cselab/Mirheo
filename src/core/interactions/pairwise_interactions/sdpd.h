#pragma once

#include "fetchers.h"
#include "pressure_EOS.h"
#include "density_kernels.h"

#include <core/interactions/accumulators/force.h>
#include <core/interactions/utils/step_random_gen.h>
#include <core/ymero_state.h>

#include <random>

class CellList;
class LocalParticleVector;

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPDHandler : public ParticleFetcherWithVelocityAndDensity
{
public:

    static constexpr float zeta = 3 + 2;
    
    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensity;
    
    PairwiseSDPDHandler(float rc, PressureEOS pressure, DensityKernel densityKernel, float viscosity, float kBT, float dt) :
        ParticleFetcherWithVelocityAndDensity(rc),
        pressure(pressure),
        densityKernel(densityKernel),
        eta(viscosity),
        fRfact(sqrt(2 * zeta * viscosity * kBT / dt))
    {
        inv_rc = 1.0 / rc;
    }

    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {        
        float3 dr = dst.p.r - src.p.r;
        float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        float pi = pressure(dst.d);
        float pj = pressure(src.d);

        float inv_disq = 1.f / (dst.d * dst.d);
        float inv_djsq = 1.f / (src.d * src.d);

        float inv_rij = rsqrtf(rij2);
        float rij = rij2 * inv_rij;
        float dWdr = densityKernel.derivative(rij, inv_rc);

        float3 er = dr * inv_rij;
        float3 du = dst.p.u - src.p.u;
        float erdotdu = dot(er, du);

        float myrandnr = Logistic::mean0var1(seed, min(src.p.i1, dst.p.i1), max(src.p.i1, dst.p.i1));

        float Aij = (inv_disq + inv_djsq) * dWdr;
        float fC = (inv_disq * pi + inv_djsq * pj) * dWdr;
        float fD = eta * Aij * zeta * inv_rij * erdotdu;
        float fR = fRfact * sqrt(Aij * inv_rij) * myrandnr;
        
        return (fC + fD + fR) * er;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    float inv_rc;
    float seed;
    PressureEOS pressure;
    DensityKernel densityKernel;
    float eta, fRfact;
};

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPD : public PairwiseSDPDHandler<PressureEOS, DensityKernel>
{
public:

    using HandlerType = PairwiseSDPDHandler<PressureEOS, DensityKernel>;
    
    PairwiseSDPD(float rc, PressureEOS pressure, DensityKernel densityKernel, float viscosity, float kBT, float dt, long seed = 42424242) :
        PairwiseSDPDHandler<PressureEOS, DensityKernel>(rc, pressure, densityKernel, viscosity, kBT, dt),
        stepGen(seed)
    {}

    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }
    
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const YmrState *state)
    {
        this->seed = stepGen.generate(state);
    }

protected:

    StepRandomGen stepGen;    
};
