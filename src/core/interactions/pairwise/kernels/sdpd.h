#pragma once

#include "accumulators/force.h"
#include "density_kernels.h"
#include "fetchers.h"
#include "interface.h"
#include "pressure_EOS.h"

#include <core/interactions/utils/step_random_gen.h>
#include <core/utils/restart_helpers.h>
#include <core/mirheo_state.h>

#include <fstream>
#include <random>

class CellList;
class LocalParticleVector;

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPDHandler : public ParticleFetcherWithVelocityDensityAndMass
{
public:
    
    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensityAndMass;
    
    PairwiseSDPDHandler(float rc, PressureEOS pressure, DensityKernel densityKernel, float viscosity, float kBT, float dt) :
        ParticleFetcherWithVelocityDensityAndMass(rc),
        inv_rc(1.0 / rc),
        pressure(pressure),
        densityKernel(densityKernel),
        fRfact(sqrt(2 * zeta * viscosity * kBT / dt)),
        fDfact(viscosity * zeta)
    {}

    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {        
        const float3 dr = dst.p.r - src.p.r;
        const float rij2 = dot(dr, dr);

        if (rij2 > rc2)
            return make_float3(0.0f);
        
        const float di = dst.d;
        const float dj = src.d;
        
        const float pi = pressure(di * dst.m);
        const float pj = pressure(dj * src.m);

        const float inv_disq = 1.f / (di * di);
        const float inv_djsq = 1.f / (dj * dj);

        const float inv_rij = rsqrtf(rij2);
        const float rij = rij2 * inv_rij;
        const float dWdr = densityKernel.derivative(rij, inv_rc);

        const float3 er = dr * inv_rij;
        const float3 du = dst.p.u - src.p.u;
        const float erdotdu = dot(er, du);

        const float myrandnr = Logistic::mean0var1(seed, min(src.p.i1, dst.p.i1), max(src.p.i1, dst.p.i1));

        const float Aij = (inv_disq + inv_djsq) * dWdr;
        const float fC = - (inv_disq * pi + inv_djsq * pj) * dWdr;
        const float fD = fDfact *        Aij * inv_rij  * erdotdu;
        const float fR = fRfact * sqrtf(-Aij * inv_rij) * myrandnr;
        
        return (fC + fD + fR) * er;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    static constexpr float zeta = 3 + 2;

    float inv_rc;
    float seed {0.f};
    PressureEOS pressure;
    DensityKernel densityKernel;
    float fDfact, fRfact;
};

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPD : public PairwiseKernel, public PairwiseSDPDHandler<PressureEOS, DensityKernel>
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
    
    void setup(__UNUSED LocalParticleVector *lpv1,
               __UNUSED LocalParticleVector *lpv2,
               __UNUSED CellList *cl1,
               __UNUSED CellList *cl2, const MirState *state) override
    {
        this->seed = stepGen.generate(state);
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
