// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/** \brief compute the gradient of a scalar field using finite differences on the device
    \tparam FieldHandler Type of device handler describing the field. Must contain parenthesis operator
    \param [in] field The functor that describes the continuous scalar field
    \param [in] x The position at which to compute the gradient
    \param [in] h The step size used to compute the gradient
    \return The approximation of the gradient of \p field at \p x
 */
template <typename FieldHandler>
inline __D__ real3 computeGradient(const FieldHandler& field, real3 x, real h)
{
    const real mx = field(x + make_real3(-h,  0,  0));
    const real px = field(x + make_real3( h,  0,  0));
    const real my = field(x + make_real3( 0, -h,  0));
    const real py = field(x + make_real3( 0,  h,  0));
    const real mz = field(x + make_real3( 0,  0, -h));
    const real pz = field(x + make_real3( 0,  0,  h));

    const real3 diff {px - mx,
                      py - my,
                      pz - mz};

    return (1.0_r / (2.0_r * h)) * diff;
}

} // namespace mirheo
