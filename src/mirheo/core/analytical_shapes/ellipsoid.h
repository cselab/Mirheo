// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/** \brief Represents an ellipsoid.

    \rst
    The ellipsoid is centered at the origin and oriented along its principal axes.
    the three radii are passed through the `axes` variable.
    The surface is described implicitly by the zero level set of:

    .. math::

        \left( \frac {x} {a_x} \right)^2 +
        \left( \frac {y} {a_y} \right)^2 +
        \left( \frac {z} {a_z} \right)^2 = 1

    \endrst
 */
class Ellipsoid
{
public:
    /** \brief Construct a \c Ellipsoid object.
        \param [in] axes the "radius" along each principal direction.
    */
    Ellipsoid(real3 axes) :
        axes_(axes),
        invAxes_(1.0 / axes)
    {}

    /**\brief Implicit surface representation.
       \param [in] r The position at which to evaluate the in/out field.
       \return The value of the field at the given position.

       This scalar field is a smooth function of the position.
       It is negative inside the ellipsoid and positive outside.
       The zero level set of that field is the surface of the ellipsoid.
    */
    __HD__ real inOutFunction(real3 r) const
    {
        return sqr(r.x * invAxes_.x) + sqr(r.y * invAxes_.y) + sqr(r.z * invAxes_.z) - 1.0_r;
    }

    /**\brief Get the normal pointing outside the ellipsoid.
       \param [in] r The position at which to evaluate the normal.
       \return The normal at r (length of this return must be 1).

       This vector field is defined everywhere in space.
       On the surface, it represents the normal vector of the surface.
    */
    __HD__ real3 normal(real3 r) const
    {
        constexpr real eps {1e-6_r};
        const real3 n {axes_.y*axes_.y * axes_.z*axes_.z * r.x,
                       axes_.z*axes_.z * axes_.x*axes_.x * r.y,
                       axes_.x*axes_.x * axes_.y*axes_.y * r.z};
        const real l = length(n);

        if (l > eps)
            return n / l;

        return {1.0_r, 0.0_r, 0.0_r}; // arbitrary if r = 0
    }

    /**\brief Get the inertia tensor of the ellipsoid in its frame of reference.
       \param [in] totalMass The total mass of the ellipsoid.
       \return The diagonal of the inertia tensor.
    */
    real3 inertiaTensor(real totalMass) const
    {
        return totalMass / 5.0_r * make_real3
            (sqr(axes_.y) + sqr(axes_.z),
             sqr(axes_.x) + sqr(axes_.z),
             sqr(axes_.x) + sqr(axes_.y));
    }

    static const char *desc;  ///< the description of shape.

private:
    real3 axes_;    ///< radii along each direction
    real3 invAxes_; ///< 1/axes
};

} // namespace mirheo
