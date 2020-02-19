#pragma once

#include "parameters.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

class LinearPressureEOS
{
public:
    using ParamsType = LinearPressureEOSParams;
    
    LinearPressureEOS(real soundSpeed, real rho0) :
        cSq_(soundSpeed * soundSpeed),
        rho0_(rho0)
    {}

    LinearPressureEOS(const ParamsType& p) :
        LinearPressureEOS(p.soundSpeed, p.rho0)
    {}

    __D__ inline real operator()(real rho) const
    {
        return cSq_ * (rho - rho0_);
    }

private:
    real cSq_;
    real rho0_;
};
MIRHEO_TYPE_NAME_AUTO(LinearPressureEOS);


class QuasiIncompressiblePressureEOS
{
public:
    using ParamsType = QuasiIncompressiblePressureEOSParams;
    
    QuasiIncompressiblePressureEOS(real p0, real rhor) :
        p0_(p0),
        invRhor_(1.0_r / rhor)
    {}

    QuasiIncompressiblePressureEOS(const ParamsType& p) :
        QuasiIncompressiblePressureEOS(p.p0, p.rhor)
    {}
    
    __D__ inline real operator()(real rho) const
    {
        const real r = rho * invRhor_;
        const real r3 = r*r*r;
        const real r7 = r3*r3*r;
        return p0_ * (r7 - 1._r);
    }

private:
    real p0_;
    real invRhor_;
};
MIRHEO_TYPE_NAME_AUTO(QuasiIncompressiblePressureEOS);

} // namespace mirheo
