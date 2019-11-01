#pragma once

#include "parameters.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>

class LinearPressureEOS
{
public:

    LinearPressureEOS(real soundSpeed, real rho0) :
        cSq(soundSpeed * soundSpeed),
        rho0(rho0)
    {}

    LinearPressureEOS(const LinearPressureEOSParams& p) :
        LinearPressureEOS(p.soundSpeed, p.rho0)
    {}
    
    __D__ inline real operator()(real rho) const
    {
        return cSq * (rho - rho0);
    }

private:
    real cSq, rho0;
};


class QuasiIncompressiblePressureEOS
{
public:
    
    QuasiIncompressiblePressureEOS(real p0, real rhor) :
        p0(p0),
        invRhor(1.0_r / rhor)
    {}

    QuasiIncompressiblePressureEOS(const QuasiIncompressiblePressureEOSParams& p) :
        QuasiIncompressiblePressureEOS(p.p0, p.rhor)
    {}
    
    __D__ inline real operator()(real rho) const
    {
        const real r = rho * invRhor;
        const real r3 = r*r*r;
        const real r7 = r3*r3*r;
        return p0 * (r7 - 1._r);
    }

private:

    real p0, invRhor;
};
