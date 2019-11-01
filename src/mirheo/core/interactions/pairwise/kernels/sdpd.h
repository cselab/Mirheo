#pragma once

#include "accumulators/force.h"
#include "density_kernels.h"
#include "fetchers.h"
#include "interface.h"
#include "pressure_EOS.h"

#include <mirheo/core/interactions/utils/step_random_gen.h>
#include <mirheo/core/utils/restart_helpers.h>
#include <mirheo/core/mirheo_state.h>

#include <fstream>
#include <random>

namespace mirheo
{

class CellList;
class LocalParticleVector;

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPDHandler : public ParticleFetcherWithVelocityDensityAndMass
{
public:
    
    using ViewType     = PVviewWithDensities;
    using ParticleType = ParticleWithDensityAndMass;
    
    PairwiseSDPDHandler(real rc, PressureEOS pressure, DensityKernel densityKernel, real viscosity, real fRfact) :
        ParticleFetcherWithVelocityDensityAndMass(rc),
        inv_rc(1.0 / rc),
        pressure(pressure),
        densityKernel(densityKernel),
        fRfact(fRfact),
        fDfact(viscosity * zeta)
    {}

    __D__ inline real3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {        
        const real3 dr = dst.p.r - src.p.r;
        const real rij2 = dot(dr, dr);

        if (rij2 > rc2)
            return make_real3(0.0_r);
        
        const real di = dst.d;
        const real dj = src.d;
        
        const real pi = pressure(di * dst.m);
        const real pj = pressure(dj * src.m);

        const real inv_disq = 1._r / (di * di);
        const real inv_djsq = 1._r / (dj * dj);

        const real inv_rij = math::rsqrt(rij2);
        const real rij = rij2 * inv_rij;
        const real dWdr = densityKernel.derivative(rij, inv_rc);

        const real3 er = dr * inv_rij;
        const real3 du = dst.p.u - src.p.u;
        const real erdotdu = dot(er, du);

        const real myrandnr = Logistic::mean0var1(seed,
                                                  math::min(static_cast<int>(src.p.i1), static_cast<int>(dst.p.i1)),
                                                  math::max(static_cast<int>(src.p.i1), static_cast<int>(dst.p.i1)));

        const real Aij = (inv_disq + inv_djsq) * dWdr;
        const real fC = - (inv_disq * pi + inv_djsq * pj) * dWdr;
        const real fD = fDfact *             Aij * inv_rij  * erdotdu;
        const real fR = fRfact * math::sqrt(-Aij * inv_rij) * myrandnr;
        
        return (fC + fD + fR) * er;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    static constexpr real zeta = 3 + 2;

    real inv_rc;
    real seed {0._r};
    PressureEOS pressure;
    DensityKernel densityKernel;
    real fDfact, fRfact;
};

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPD : public PairwiseKernel, public PairwiseSDPDHandler<PressureEOS, DensityKernel>
{
public:

    using HandlerType = PairwiseSDPDHandler<PressureEOS, DensityKernel>;
    
    PairwiseSDPD(real rc, PressureEOS pressure, DensityKernel densityKernel, real viscosity, real kBT, real dt, long seed = 42424242) :
        PairwiseSDPDHandler<PressureEOS, DensityKernel>(rc, pressure, densityKernel, viscosity, computeFRfact(viscosity, kBT, dt)),
        stepGen(seed),
        viscosity(viscosity),
        kBT(kBT)
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
        this->fRfact = computeFRfact(this->viscosity, this->kBT, state->dt);
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

    static real computeFRfact(real viscosity, real kBT, real dt)
    {
        return math::sqrt(2 * HandlerType::zeta * viscosity * kBT / dt);
    }
    
    StepRandomGen stepGen;
    real viscosity;
    real kBT;
};

} // namespace mirheo
