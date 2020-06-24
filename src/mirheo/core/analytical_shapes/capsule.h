// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/** \brief Represents a capsule.

    A capsule is represented by a segment and a radius.
    Its surfae is the set of points whose distance to the
    segment is equal to the radius.

    In more visual terms, a capsule looks like a finite cylinder
    with two half spheres on its ends.

    The capsule is centered at the origin and oriented along the z axis.
 */
class Capsule
{
public:
    /** \brief Construct a \c Capsule.
        \param [in] R the radius of the capsule.  Must be positive.
        \param [in] L the length of the segment used to represent the
                      capsule. Must be positive.
    */
    Capsule(real R, real L) :
        R_(R),
        halfL_(0.5_r * L)
    {}

    /**\brief Implicit surface representation.
       \param [in] r The position at which to evaluate the in/out field.
       \return The value of the field at the given position.

       This scalar field is a smooth function of the position.
       It is negative inside the capsule and positive outside.
       The zero level set of that field is the surface of the capsule.
    */
    __HD__ real inOutFunction(real3 r) const
    {
        const real dz = math::abs(r.z) - halfL_;

        real drsq = sqr(r.x) + sqr(r.y);
        if (dz > 0) drsq += sqr(dz);

        const real dr = math::sqrt(drsq) - R_;
        return dr;
    }

    /**\brief Get the normal pointing outside the capsule.
       \param [in] r The position at which to evaluate the normal.
       \return The normal at r (length of this return must be 1).

       This vector field is defined everywhere in space.
       On the surface, it represents the normal vector of the surface.
    */
    __HD__ real3 normal(real3 r) const
    {
        constexpr real eps = 1e-6_r;

        const real dz = math::abs(r.z) - halfL_;

        real rsq = sqr(r.x) + sqr(r.y);
        if (dz > 0) rsq += sqr(dz);

        const real rinv = rsq > eps ? math::rsqrt(rsq) : 0._r;

        const real3 n {r.x,
                       r.y,
                       dz > 0 ? dz : 0._r};
        return rinv * n;
    }


    /**\brief Get the inertia tensor of the capsule in its frame of reference.
       \param [in] totalMass The total mass of the capsule.
       \return The diagonal of the inertia tensor.
    */
    real3 inertiaTensor(real totalMass) const
    {
        const real R2 = R_ * R_;
        const real R3 = R2 * R_;
        const real R4 = R2 * R2;
        const real R5 = R3 * R2;

        const real V_pi   = 2.0_r * halfL_ * R2 + (4.0_r / 3.0_r) * R3;

        const real xxB_pi = R5 * (4.0_r / 15.0_r);
        const real xxC_pi = R4 * halfL_ * 0.5_r;

        const real zzB_pi = 4.0_r * (halfL_ * halfL_ * R3 / 3.0_r
                                     + halfL_ * R4 / 4.0_r
                                     + R5 / 15.0_r);
        const real zzC_pi = R2 * halfL_ * halfL_ * halfL_ * (2.0_r / 3.0f);

        const real xx = totalMass * (xxB_pi + xxC_pi) / V_pi;
        const real zz = totalMass * (zzB_pi + zzC_pi) / V_pi;
        const real yy = xx;

        return make_real3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;  ///< the description of shape.

private:
    real R_;     ///< radius
    real halfL_; ///< half length between the two sphere centers
};

} // namespace mirheo
