// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "parameters.h"

#include <cmath>

namespace mirheo
{

/// Density kernel used in MDPD
class SimpleMDPDDensityKernel
{
public:
    /// parameters struct for this type
    using ParamsType = SimpleMDPDDensityKernelParams;

    SimpleMDPDDensityKernel() = default;
    /// generic constructor
    SimpleMDPDDensityKernel(const ParamsType&) {};

    /// Kernel value at r
    __D__ inline real operator()(real r, real invrc) const
    {
        const real rm = (1._r - r * invrc) * invrc;

        return normalization_ * rm * rm * invrc;
    }
private:
    static constexpr real normalization_ = 15.0 / (2.0 * M_PI);
};
/// create type name
MIRHEO_TYPE_NAME_AUTO(SimpleMDPDDensityKernel);

/// Density kernel from Wendland C2 function
class WendlandC2DensityKernel
{
public:
     /// parameters struct for this type
    using ParamsType = WendlandC2DensityKernelParams;

    WendlandC2DensityKernel() = default;
    /// generic constructor
    WendlandC2DensityKernel(const ParamsType&) {};

    /// Kernel value at r
    __D__ inline real operator()(real r, real invrc) const
    {
        const real r_ = r * invrc;
        const real rm = 1.0_r - r_;
        const real rm2 = rm * rm;
        const real invrc3 = invrc * invrc * invrc;

        return normalization_ * invrc3 * rm2 * rm2 * (1.0_r + 4.0_r * r_);
    }

    /// The derivative at r (needed by PairwiseSDPDHandler)
    __D__ inline real derivative(real r, real invrc) const
    {
        const real r_ = r * invrc;
        const real rm = r_ - 1._r;
        const real invrc3 = invrc * invrc * invrc;
        return 20.0_r * invrc3 * normalization_ * r_ * rm*rm*rm * invrc;
    }

private:
    static constexpr real normalization_ = 21.0 / (2.0 * M_PI);
};
/// create type name
MIRHEO_TYPE_NAME_AUTO(WendlandC2DensityKernel);


} // namespace mirheo
