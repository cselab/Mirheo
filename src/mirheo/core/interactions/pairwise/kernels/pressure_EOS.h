// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "parameters.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

/// Compute pressure from density
/// with a linear equation of state
class LinearPressureEOS
{
public:
    using ParamsType = LinearPressureEOSParams; ///< The corresponding parameters

    /// Constructor
    LinearPressureEOS(real soundSpeed, real rho0) :
        cSq_(soundSpeed * soundSpeed),
        rho0_(rho0)
    {}

    /// Generic constructor
    LinearPressureEOS(const ParamsType& p) :
        LinearPressureEOS(p.soundSpeed, p.rho0)
    {}

    /// Compute the pressure from mass density
    __D__ inline real operator()(real rho) const
    {
        return cSq_ * (rho - rho0_);
    }

private:
    real cSq_;  ///< speed of sound squared
    real rho0_; ///< reference mass density
};
/// set type name
MIRHEO_TYPE_NAME_AUTO(LinearPressureEOS);


/// Compute pressure from density
/// with the "quasi-incompressible" equation of state
class QuasiIncompressiblePressureEOS
{
public:
    using ParamsType = QuasiIncompressiblePressureEOSParams; ///< The corresponding parameters

    /// Constructor
    QuasiIncompressiblePressureEOS(real p0, real rhor) :
        p0_(p0),
        invRhor_(1.0_r / rhor)
    {}

    /// Generic constructor
    QuasiIncompressiblePressureEOS(const ParamsType& p) :
        QuasiIncompressiblePressureEOS(p.p0, p.rhor)
    {}

    /// Compute the pressure from mass density
    __D__ inline real operator()(real rho) const
    {
        const real r = rho * invRhor_;
        const real r3 = r*r*r;
        const real r7 = r3*r3*r;
        return p0_ * (r7 - 1._r);
    }

private:
    real p0_;      ///< pressure magnitude
    real invRhor_; ///< inverse of reference mass density
};
/// set type name
MIRHEO_TYPE_NAME_AUTO(QuasiIncompressiblePressureEOS);

} // namespace mirheo
