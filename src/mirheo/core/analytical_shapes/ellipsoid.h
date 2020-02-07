#pragma once

#include "interface.h"

#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/** \brief Represents an ellipsoid.

    \rst
    The ellipsoid is centered at the origin and oriented along its principal axes.
    the three radii are passed through the `axes` variable. 
    The surface is described by:

    .. math::

        \left( \frac {x} {a_x} \right)^2 +
        \left( \frac {y} {a_y} \right)^2 +
        \left( \frac {z} {a_z} \right)^2 = 1

    \endrst
 */
class Ellipsoid : public AnalyticShape
{
public:
    /** \brief Construct a \c Ellipsoid object.
        \param [in] axes the "radius" along each principal direction.
    */
    Ellipsoid(real3 axes) :
        axes_(axes),
        invAxes_(1.0 / axes)
    {}

    __HD__ real inOutFunction(real3 r) const override
    {
        return sqr(r.x * invAxes_.x) + sqr(r.y * invAxes_.y) + sqr(r.z * invAxes_.z) - 1.0_r;
    }

    __HD__ real3 normal(real3 r) const override
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
    
    real3 inertiaTensor(real totalMass) const override
    {
        return totalMass / 5.0_r * make_real3
            (sqr(axes_.y) + sqr(axes_.z),
             sqr(axes_.x) + sqr(axes_.z),
             sqr(axes_.x) + sqr(axes_.y));
    }

    static const char *desc;  ///< the description of shape.
    
private:    
    real3 axes_, invAxes_;
};

} // namespace mirheo
