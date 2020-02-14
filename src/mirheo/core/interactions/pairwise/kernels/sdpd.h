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
        invrc_(1.0 / rc),
        pressure_(pressure),
        densityKernel_(densityKernel),
        fRfact_(fRfact),
        fDfact_(viscosity * zeta_)
    {}

    __D__ inline real3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        constexpr real eps = 1e-6_r;
        const real3 dr = dst.p.r - src.p.r;
        const real rij2 = dot(dr, dr);

        if (rij2 > rc2_ || rij2 < eps)
            return make_real3(0.0_r);
        
        const real di = dst.d;
        const real dj = src.d;
        
        const real pi = pressure_(di * dst.m);
        const real pj = pressure_(dj * src.m);

        const real inv_disq = 1._r / (di * di);
        const real inv_djsq = 1._r / (dj * dj);

        const real inv_rij = math::rsqrt(rij2);
        const real rij = rij2 * inv_rij;
        const real dWdr = densityKernel_.derivative(rij, invrc_);

        const real3 er = dr * inv_rij;
        const real3 du = dst.p.u - src.p.u;
        const real erdotdu = dot(er, du);

        const real myrandnr = Logistic::mean0var1(seed_,
                                                  math::min(static_cast<int>(src.p.i1), static_cast<int>(dst.p.i1)),
                                                  math::max(static_cast<int>(src.p.i1), static_cast<int>(dst.p.i1)));

        const real Aij = (inv_disq + inv_djsq) * dWdr;
        const real Aij_rij = math::min(-0.0_r, Aij * inv_rij); // must be negative because of sqrt below
        
        const real fC = - (inv_disq * pi + inv_djsq * pj) * dWdr;
        const real fD = fDfact_ *             Aij_rij  * erdotdu;
        const real fR = fRfact_ * math::sqrt(-Aij_rij) * myrandnr;
        
        return (fC + fD + fR) * er;
    }

    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

protected:

    static constexpr real zeta_ = 3 + 2;

    real invrc_;
    real seed_ {0._r};
    PressureEOS pressure_;
    DensityKernel densityKernel_;
    real fDfact_, fRfact_;
};

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPD : public PairwiseKernel, public PairwiseSDPDHandler<PressureEOS, DensityKernel>
{
public:

    using HandlerType = PairwiseSDPDHandler<PressureEOS, DensityKernel>;
    
    PairwiseSDPD(real rc, PressureEOS pressure, DensityKernel densityKernel, real viscosity, real kBT, real dt, long seed = 42424242) :
        PairwiseSDPDHandler<PressureEOS, DensityKernel>(rc, pressure, densityKernel, viscosity, computeFRfact(viscosity, kBT, dt)),
        stepGen_(seed),
        viscosity_(viscosity),
        kBT_(kBT)
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
        this->seed_ = stepGen_.generate(state);
        this->fRfact_ = computeFRfact(this->viscosity_, this->kBT_, state->dt);
    }

    void writeState(std::ofstream& fout) override
    {
        TextIO::writeToStream(fout, stepGen_);
    }

    bool readState(std::ifstream& fin) override
    {
        return TextIO::readFromStream(fin, stepGen_);
    }
    
    
protected:

    static real computeFRfact(real viscosity, real kBT, real dt)
    {
        return math::sqrt(2 * HandlerType::zeta_ * viscosity * kBT / dt);
    }
    
    StepRandomGen stepGen_;
    real viscosity_;
    real kBT_;
};

} // namespace mirheo
