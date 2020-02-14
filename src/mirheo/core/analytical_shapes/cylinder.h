#pragma once

#include "interface.h"

#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/** \brief Represents a finite cylinder.

    The cylinder is centered at the origin and is oriented along the z axis.
    It is fully described by its length and its radius.
 */
class Cylinder: public AnalyticShape
{
public:
    /** \brief Constructs a \c Cylinder.
        \param [in] R the radius of the cylinder. Must be positive
        \param [in] L the length of the cylinder. Must be positive.
    */
    Cylinder(real R, real L) :
        R_(R),
        halfL_(0.5_r * L)
    {}

    __HD__ inline real inOutFunction(real3 coo) const override
    {
        const real dr = math::sqrt(sqr(coo.x) + sqr(coo.y)) - R_;
        const real dz = math::abs(coo.z) - halfL_;

        const real dist2edge = math::sqrt(sqr(dz) + sqr(dr));
        const real dist2disk = dr > 0 ? dist2edge : dz;
        const real dist2cyl  = dz > 0 ? dist2edge : dr;

        return (dz <= 0) && (dr <= 0)
            ? math::max(dist2cyl, dist2disk)
            : math::min(dist2cyl, dist2disk);
    }

    __HD__ inline real3 normal(real3 coo) const override
    {
        constexpr real eps   = 1e-6_r;
        constexpr real delta = 1e-3_r;
        
        const real rsq = sqr(coo.x) + sqr(coo.y);
        const real rinv = rsq > eps ? math::rsqrt(rsq) : 0._r;

        const real dr = math::sqrt(rsq) - R_;
        const real dz = math::abs(coo.z) - halfL_;
        
        const real3 er {rinv * coo.x, rinv * coo.y, 0._r};
        const real3 ez {0._r, 0._r, coo.z > 0 ? 1._r : -1._r};

        
        real3 n {0._r, 0._r, 0._r};
        if (math::abs(dr) < delta) n += er;
        if (math::abs(dz) < delta) n += ez;
        return n;
    }
    

    inline real3 inertiaTensor(real totalMass) const override
    {
        const real xx = totalMass * R_ * R_ * 0.25_r;
        const real yy = xx;
        const real zz = totalMass * halfL_ * halfL_ * 0.3333333_r;
        
        return make_real3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;  ///< the description of shape.
    
private:
    
    real R_, halfL_;
};

} // namespace mirheo
